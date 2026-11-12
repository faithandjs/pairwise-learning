"""Step 10: synthetic choices: simulated users pick between pairs of candidate routes, with noise.

Run:  python step10_choices.py
Reads:  outputs/candidates.csv
Writes: outputs/choices.csv, outputs/normalisation.json

For every profile, every start/end pair and every pair of its candidate routes (i, j):
  features are rescaled to 0-1 with fixed study-area reference values (the largest value of each feature seen on any
  candidate route; saved in normalisation.json and kept the same for training and testing);
  utility = minus (sum of weight x feature);  difference du = utility(i) - utility(j);
  the simulated user picks i with probability 1 / (1 + exp(-du / noise)).
A smaller noise value means a more consistent user. Three noise levels are produced (methodology point 1).
"""
import json

import numpy as np
import pandas as pd

import config
from common import ban_key
from profiles import PROFILES

NOISE = getattr(config, "CHOICE_NOISE_LEVELS", {
                # Note-> my values; change in config.py
                "low": 0.02, "medium": 0.05, "high": 0.10})
SEED = getattr(config, "RANDOM_SEED", 42)
FEATURES = {"distance": "distance_m", "gradient": "steep_m",
            "steps": "steps_m", "kerb": "kerb_units"}


def main():
    cand = pd.read_csv(config.OUTPUTS / "candidates.csv")
    ref = {f: (float(cand[c].max()) or 1.0) for f, c in FEATURES.items()}
    (config.OUTPUTS / "normalisation.json").write_text(json.dumps(ref, indent=2))
    print("reference values (feature = 1.0):", {
          f: round(v, 1) for f, v in ref.items()})
    for f, c in FEATURES.items():
        cand["x_" + f] = (cand[c] / ref[f]).clip(upper=1.0)
    xcols = ["x_" + f for f in FEATURES]

    rows = []
    for pid, p in PROFILES.items():
        w = np.array([p["weights"].get(f, 0.0) for f in FEATURES])
        g = cand[cand["ban_set"] == ban_key(p["bans"])]
        for od_id, grp in g.groupby("od_id"):
            X, ids = grp[xcols].to_numpy(), grp["route_id"].to_numpy()
            for i in range(len(grp)):
                for j in range(i + 1, len(grp)):
                    d = X[i] - X[j]
                    rows.append(
                        (pid, od_id, ids[i], ids[j], *d, -float(w @ d)))
    base = pd.DataFrame(rows, columns=["profile", "od_id", "route_i", "route_j"]
                        + ["dx_" + f for f in FEATURES] + ["du_true"])

    rng = np.random.default_rng(SEED)
    parts = []
    for name, tau in NOISE.items():
        part = base.copy()
        part["noise"], part["noise_tau"] = name, tau
        p_i = 1.0 / (1.0 + np.exp(-part["du_true"] / tau))
        part["chose_i"] = (rng.random(len(part)) < p_i).astype(int)
        part["true_prefers_i"] = np.where(
            part["du_true"] > 0, 1, np.where(part["du_true"] < 0, 0, np.nan))
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(config.OUTPUTS / "choices.csv", index=False)

    rep = []
    for pid, g in base.groupby("profile"):
        row = {"profile": f"{pid} {PROFILES[pid]['name']}", "comparisons": len(g),
               "utility_differs_%": round(100 * (g["du_true"] != 0).mean(), 0)}
        for f in FEATURES:
            row[f"{f}_differs_%"] = round(100 * (g["dx_" + f] != 0).mean(), 0)
        for name in NOISE:
            o = out[(out.profile == pid) & (out.noise == name)
                    & out.true_prefers_i.notna()]
            row[f"agrees_with_best_{name}_%"] = round(
                100 * (o.chose_i == o.true_prefers_i).mean(), 0) if len(o) else np.nan
        rep.append(row)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print()
        print(pd.DataFrame(rep).to_string(index=False))
    print(f"\n{len(base):,} comparisons x {len(NOISE)} noise levels = {len(out):,} rows saved to outputs/choices.csv")


if __name__ == "__main__":
    main()
