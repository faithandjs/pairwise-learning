"""Step 11: the preference learner (Bradley-Terry / logistic pairwise preference learning) and its evaluation on candidates.

Run:  python step11_learner.py
Reads:  outputs/candidates.csv
Writes: outputs/learner_results.csv  (one row per repeat x profile x noise level x share of training data)
        outputs/learner_weights.csv  (true and learned weights, per feature)

One REPEAT (seed s) does this:
  1. Split the origin-destination pairs into train / validation / test (60/20/20 by default). Splitting is by PAIR,
     so no route from a test journey is ever seen in training.
  2. Work out the reference values (largest value of each feature) from the TRAINING pairs only, and rescale
     all routes with them. (This is change 2: the test pairs do not influence the scale.)
  3. For each profile and noise level, generate the simulated choices with a random draw seeded by (s, profile, noise).
     (This is change 1: the choices are generated inside the repeat.)
  4. Fit logistic regression, with no intercept, on the feature differences between the two routes.
     The strength of regularisation (C) is chosen on the validation pairs.
  5. Score on the test pairs, against the same data for three baselines:
        oracle   = the true weights that generated the choices
        generic  = equal weights on the four features
        shortest = prefers the shorter route
Everything is repeated for several seeds and for several shares of the training data (the learning curve).

The learned model is  P(route i chosen over j) = sigmoid(beta . (x_i - x_j)),  where x = rescaled features.
Because the true model is  sigmoid(-(w/noise) . (x_i - x_j)), beta should be about -w/noise: the learned penalty
weights are -beta, and only their RELATIVE sizes can be recovered (the noise level and the weights scale together),
so weights are compared after rescaling to sum to 1, over the features that actually vary for that profile.
"""
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import config
from common import ban_key, split_journeys
from profiles import PROFILES

warnings.filterwarnings("ignore")

FEATURES = ["distance", "gradient", "steps", "kerb"]
COLS = ["distance_m", "steep_m", "steps_m", "kerb_units"]
NOISE = getattr(config, "CHOICE_NOISE_LEVELS", {"low": 0.02, "medium": 0.05, "high": 0.10})
N_REPEATS = getattr(config, "N_REPEATS", 10)                       # Note-> seeds 0 to N_REPEATS-1
SPLIT = getattr(config, "SPLIT_FRACTIONS", (0.6, 0.2, 0.2))        # Note-> train, validation, test
TRAIN_FRACTIONS = getattr(config, "TRAIN_FRACTIONS", (0.1, 0.25, 0.5, 1.0))   # Note-> learning curve
C_GRID = getattr(config, "C_GRID", (0.01, 0.1, 1, 10, 100, 1000))  # Note-> regularisation strengths tried
GENERIC = np.full(4, 0.25)                                         # Note-> generic baseline: equal weights


class Cands:
    """Candidate routes of one ban set, as arrays, plus every pair of routes within the same journey."""

    def __init__(self, g):
        g = g.sort_values(["od_id", "route_id"]).reset_index(drop=True)
        self.R = g[COLS].to_numpy(float)
        self.od = g["od_id"].to_numpy()
        self.groups = {o: idx for o, idx in g.groupby("od_id").indices.items()}
        I, J, OD = [], [], []
        for o, idx in self.groups.items():
            if len(idx) < 2:
                continue
            a, b = np.triu_indices(len(idx), 1)
            I.append(idx[a]); J.append(idx[b]); OD.append(np.full(len(a), o))
        empty = np.array([], dtype=int)
        self.I, self.J, self.OD = (np.concatenate(v) if v else empty for v in (I, J, OD))


def accuracy(d, y):
    """Share of comparisons where the sign of d (score of i minus score of j) matches the choice y; ties count as half."""
    return float(np.mean(np.where(d > 0, y == 1, np.where(d < 0, y == 0, 0.5))))


def select(S, X, ods, w, beta):
    """For each journey pick the route each method prefers; return mean regret (true utility lost) and top-1 rate."""
    reg = {k: [] for k in ("learned", "generic", "shortest")}
    top = {k: [] for k in reg}
    for o in ods:
        idx = S.groups.get(o)
        if idx is None or len(idx) < 2:
            continue
        u = -(X[idx] @ w)
        scores = {"learned": X[idx] @ beta, "generic": -(X[idx] @ GENERIC), "shortest": -S.R[idx, 0]}
        for k, s in scores.items():
            c = int(np.argmax(s))
            reg[k].append(u.max() - u[c])
            top[k].append(float(u[c] == u.max()))
    return ({k: float(np.mean(v)) if v else np.nan for k, v in reg.items()},
            {k: float(np.mean(v)) if v else np.nan for k, v in top.items()})


def main():
    cand = pd.read_csv(config.OUTPUTS / "candidates.csv")
    sets = {key: Cands(g) for key, g in cand.groupby("ban_set")}
    all_ods = np.array(sorted(cand["od_id"].unique()))
    rows, wrows = [], []
    skipped = set()

    for seed in range(N_REPEATS):
        train, val, test = split_journeys(all_ods, seed, SPLIT)

        # Note-> reference values come from the TRAINING pairs only
        ref = np.max([S.R[np.isin(S.od, train)].max(axis=0) for S in sets.values()], axis=0)
        ref = np.where(ref > 0, ref, 1.0)

        for pid, p in PROFILES.items():
            S = sets[ban_key(p["bans"])]
            X = np.minimum(S.R / ref, 1.0)
            dX = X[S.I] - X[S.J]
            w = np.array([p["weights"].get(f, 0.0) for f in FEATURES])
            du = -(dX @ w)
            m_tr, m_va, m_te = (np.isin(S.OD, g) for g in (train, val, test))
            if not m_tr.any():
                skipped.add(pid)
                continue
            varying = dX[m_tr].std(axis=0) > 1e-12
            if not varying.any():
                continue
            w_true = w[varying] / w[varying].sum()

            for ni, (nname, tau) in enumerate(NOISE.items()):
                y = (np.random.default_rng([seed, pid, ni]).random(len(du)) < 1 / (1 + np.exp(-du / tau))).astype(int)
                order = np.random.default_rng([seed, pid, ni, 99]).permutation(np.where(m_tr)[0])
                te = np.where(m_te)[0]
                for frac in TRAIN_FRACTIONS:
                    sub = order[:max(int(np.ceil(frac * len(order))), 2)]
                    if len(np.unique(y[sub])) < 2:
                        continue
                    best = None
                    t0 = time.perf_counter()
                    for C in C_GRID:
                        m = LogisticRegression(C=C, fit_intercept=False, max_iter=1000).fit(dX[sub], y[sub])
                        if m_va.any() and len(np.unique(y[m_va])) == 2:
                            loss = log_loss(y[m_va], m.predict_proba(dX[m_va])[:, 1], labels=[0, 1])
                        else:
                            loss = log_loss(y[sub], m.predict_proba(dX[sub])[:, 1], labels=[0, 1])
                        if best is None or loss < best[0]:
                            best = (loss, C, m.coef_[0].copy())
                    _, C, beta = best
                    fit_seconds = time.perf_counter() - t0

                    learned = np.clip(-beta[varying], 0, None)
                    w_hat = learned / learned.sum() if learned.sum() > 0 else np.full(varying.sum(), np.nan)
                    reg, top = select(S, X, test, w, beta)
                    row = dict(seed=seed, profile=pid, profile_name=p["name"], noise=nname, tau=tau, train_frac=frac,
                               n_train_comparisons=len(sub), n_test_comparisons=len(te), n_test_journeys=len(test),
                               C=C, fit_seconds=fit_seconds, varying=",".join(f for f, v in zip(FEATURES, varying) if v),
                               weight_mae=float(np.nanmean(np.abs(w_hat - w_true))),
                               acc_learned=accuracy(dX[te] @ beta, y[te]),
                               acc_oracle=accuracy(du[te], y[te]),
                               acc_generic=accuracy(-(dX[te] @ GENERIC), y[te]),
                               acc_shortest=accuracy(-(S.R[S.I[te], 0] - S.R[S.J[te], 0]), y[te]),
                               acc_learned_vs_truth=accuracy(dX[te] @ beta, (du[te] > 0).astype(int)),
                               regret_learned=reg["learned"], regret_generic=reg["generic"], regret_shortest=reg["shortest"],
                               top1_learned=top["learned"], top1_generic=top["generic"], top1_shortest=top["shortest"])
                    for f, r in zip(FEATURES, ref):
                        row["ref_" + f] = float(r)
                    rows.append(row)
                    if frac == max(TRAIN_FRACTIONS):
                        vf = [f for f, v in zip(FEATURES, varying) if v]
                        for f, a, b in zip(vf, w_true, w_hat):
                            wrows.append(dict(seed=seed, profile=pid, noise=nname, feature=f, true_weight=a, learned_weight=b))

    res = pd.DataFrame(rows)
    res.to_csv(config.OUTPUTS / "learner_results.csv", index=False)
    pd.DataFrame(wrows).to_csv(config.OUTPUTS / "learner_weights.csv", index=False)
    print(f"{N_REPEATS} repeats, split by journey {SPLIT}, noise {dict(NOISE)}, training shares {list(TRAIN_FRACTIONS)}")
    print("reference values (mean over repeats):", ", ".join(f"{f} {res['ref_' + f].mean():.1f}" for f in FEATURES))
    if skipped:
        print(f"profiles skipped (no pairs of candidate routes to compare): {sorted(skipped)}")

    full = res[res.train_frac == max(TRAIN_FRACTIONS)]
    show = ["acc_learned", "acc_oracle", "acc_generic", "acc_shortest", "weight_mae",
            "regret_learned", "regret_generic", "regret_shortest", "top1_learned", "top1_generic", "top1_shortest"]
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.3f}".format):
        for nname in NOISE:
            t = full[full.noise == nname].groupby(["profile", "profile_name"])[show].mean().reset_index()
            print(f"\nAll training data, noise = {nname}: mean over {N_REPEATS} repeats, test journeys only")
            print(t.to_string(index=False))
        print("\nLearning curve (mean over profiles and repeats): accuracy on test choices and weight error")
        lc = res.groupby(["noise", "train_frac"])[["acc_learned", "acc_oracle", "acc_generic", "acc_shortest", "weight_mae"]].mean()
        print(lc.reset_index().to_string(index=False))
    print("\nsaved outputs/learner_results.csv and outputs/learner_weights.csv")


if __name__ == "__main__":
    main()
