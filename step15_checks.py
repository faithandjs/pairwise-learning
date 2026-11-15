"""Step 15: two validity checks asked for after review (no new experiment, reads existing outputs).

Run:  python step15_checks.py
Reads: outputs/routing_results.csv, outputs/learner_results.csv   (run steps 11 and 12 first)

Check A, sign conventions: a tiny deterministic example in which route A is strictly better than route B on every feature.
  Prints what the simulated choice, the fitted coefficient, the learned penalty weight and the pick of the learned model do.
  Expected: choice favours A, coefficients negative, penalty weights positive, learned model picks A.

Check B, the cap at 1: the reference value of each feature is the training maximum, and any value above it is capped at 1
  when a route is SCORED (utility, regret) and when the learner is TRAINED. The routing cost itself is NOT capped.
  This check reports, for every method on the test journeys:
    - how often a whole-network route exceeds a reference value, by feature;
    - how much the cap changes mean regret (capped vs uncapped utility);
    - how often the cap makes a method look exactly as good as the oracle when it is not (hidden difference).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import config
from profiles import PROFILES

FEATURES = ["distance", "gradient", "steps", "kerb"]
COLS = {"distance": "distance_m", "gradient": "steep_m", "steps": "steps_m", "kerb": "kerb_units"}


def check_signs():
    print("CHECK A: sign conventions on a deterministic example")
    w = np.array([0.4, 0.3, 0.2, 0.1])                       # true penalty weights
    tau = 0.05
    rng = np.random.default_rng(0)
    xa = np.array([0.2, 0.1, 0.0, 0.1])                      # route A: lower (better) on every feature
    xb = np.array([0.6, 0.5, 0.4, 0.5])                      # route B: higher (worse) on every feature
    # many comparisons with random better/worse feature pairs so the model has something to fit
    X = rng.random((400, 2, 4))
    X[:, 1, :] = np.clip(X[:, 0, :] + rng.random((400, 4)) * 0.3 - 0.1, 0, 1)
    dX = X[:, 0, :] - X[:, 1, :]                              # route i minus route j, as in step11
    du = -(dX @ w)                                            # utility of i minus utility of j
    y = (rng.random(len(du)) < 1 / (1 + np.exp(-du / tau))).astype(int)   # 1 = i chosen
    m = LogisticRegression(C=100, fit_intercept=False, max_iter=1000).fit(dX, y)
    beta = m.coef_[0]
    learned = np.clip(-beta, 0, None)
    print("  A is better than B on every feature: utility(A) - utility(B) =", round(float(-(w @ (xa - xb))), 3), "(positive = A better)")
    print("  probability the simulated user chooses A:", round(float(1 / (1 + np.exp(-(-(w @ (xa - xb))) / tau))), 4))
    print("  fitted coefficients beta (expected negative, about -w/tau):", np.round(beta, 1), "| -w/tau =", np.round(-w / tau, 1))
    print("  learned penalty weights (-beta, clipped at 0, rescaled to sum 1):", np.round(learned / learned.sum(), 3), "| true:", w)
    print("  learned model score for A:", round(float(xa @ beta), 2), " for B:", round(float(xb @ beta), 2), "(higher score = preferred)")
    ok = (beta < 0).all() and (xa @ beta > xb @ beta)
    print("  RESULT:", "signs consistent" if ok else "SIGN PROBLEM: investigate")


def check_cap():
    print("\nCHECK B: effect of the cap at 1 on whole-network routes")
    rr = pd.read_csv(config.OUTPUTS / "routing_results.csv")
    rr["noise"] = rr["noise"].fillna("")                         # an empty noise level is read back from csv as NaN
    lr = pd.read_csv(config.OUTPUTS / "learner_results.csv")
    refs = lr.groupby("seed")[["ref_" + f for f in FEATURES]].first()
    d = rr[rr.get("scored", True) == True].copy()                    # noqa: E712
    d = d[(d.noise == "") | (d.noise == getattr(config, "SUMMARY_NOISE", "medium"))]
    ratio = {f: d[COLS[f]].to_numpy(float) / refs.loc[d.seed, "ref_" + f].to_numpy(float) for f in FEATURES}
    for f in FEATURES:
        d["r_" + f] = ratio[f]
    wt = {pid: np.array([p["weights"].get(f, 0.0) for f in FEATURES]) for pid, p in PROFILES.items()}
    R = d[["r_" + f for f in FEATURES]].to_numpy()
    W = np.vstack([wt[p] for p in d.profile])
    d["u_capped"] = -(W * np.minimum(R, 1)).sum(axis=1)
    d["u_uncapped"] = -(W * R).sum(axis=1)

    print("  share of routes with a feature above its reference value (%), by method:")
    t = d.groupby("method")[["r_" + f for f in FEATURES]].agg(lambda s: 100 * (s > 1).mean()).round(1)
    t.columns = FEATURES
    print(t.to_string())
    d["any_over"] = (R > 1).any(axis=1)
    print("  share of routes with ANY feature above reference: " + ", ".join(
        f"{m} {100 * g.any_over.mean():.1f}%" for m, g in d.groupby("method")))

    key = ["seed", "profile", "od_id"]                           # oracle rows carry no noise level, so match without it
    orc = d[d.method == "oracle"].set_index(key)[["u_capped", "u_uncapped"]]
    rows = []
    for m, g in d[d.method != "oracle"].groupby("method"):
        j = g.join(orc, on=key, rsuffix="_orc")
        rc = j.u_capped_orc - j.u_capped
        ru = j.u_uncapped_orc - j.u_uncapped
        rows.append(dict(method=m, n=len(j), regret_capped=rc.mean(), regret_uncapped=ru.mean(),
                         pct_hidden_by_cap=100 * ((rc.abs() < 1e-9) & (ru.abs() > 1e-9)).mean()))
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print("\n  mean regret with and without the cap (same routes), and the share of journeys where the cap hides a difference:")
        print(pd.DataFrame(rows).to_string(index=False))
    print("\n  Reading: if few routes exceed a reference and the two regrets are close, the cap does not change conclusions.")
    print("  Note: routing cost is uncapped, so route CHOICE is not affected by the cap; only training and scoring are.")


if __name__ == "__main__":
    check_signs()
    try:
        check_cap()
    except FileNotFoundError as e:
        print("\nCHECK B skipped, missing file:", e.filename, "(run steps 11 and 12)")
