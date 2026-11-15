"""Step 14: sensitivity runs. Re-run steps 8, 9, 11 and 12 with one setting changed at a time, and compare.

Run:  python step14_sensitivity.py              (runs every scenario below; it can take a while)
      python step14_sensitivity.py --redo       (run them all again, even those already finished)
Needs:  the config.py block that lets this script pass settings to the other steps (see the bottom of config.py).
Writes: outputs/sensitivity/<scenario>/...   (a full copy of the outputs of each scenario; your main outputs are untouched)
        outputs/sensitivity/summary.csv      (one row per scenario)

Each scenario changes ONE thing against the baseline (your current config.py):
    detour_1.3 / detour_2.0       drop candidate routes longer than 1.3x / 2.0x the shortest (baseline 1.5x)
    overlap_0.5 / overlap_0.7     a candidate may share at most 50% / 70% of its length with another (baseline 60%)
    unknown_not_structures        bridges and tunnels with unknown gradient stay usable for gradient-limit profiles
The ORS baseline (step 13) is not rerun: it does not depend on these settings (and the journeys change with some of them).
A scenario that is already finished is skipped, so you can stop and start again.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config

SCENARIOS = {                                   # Note-> add or edit scenarios here
    "baseline": {},
    "detour_1.3": {"MAX_ROUTE_DETOUR": 1.3},
    "detour_2.0": {"MAX_ROUTE_DETOUR": 2.0},
    "overlap_0.5": {"MAX_ROUTE_OVERLAP": 0.5},
    "overlap_0.7": {"MAX_ROUTE_OVERLAP": 0.7},
    "unknown_not_structures": {"UNKNOWN_GRADIENT_BAN": "not_structures"},
}
STEPS = ["step8_od_pairs.py", "step9_candidates.py", "step11_learner.py", "step12_routing.py"]
ROOT = Path(config.ROOT)
BASE_OUT = "outputs/sensitivity"


def run(name, over):
    rel = f"{BASE_OUT}/{name}"
    out = ROOT / rel
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CONFIG_OVERRIDES=json.dumps({**over, "OUTPUTS": rel}))
    with open(out / "log.txt", "w") as log:
        for s in STEPS:
            print(f"   {s} ...", flush=True)
            r = subprocess.run([sys.executable, s], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if r.returncode:
                raise SystemExit(f"{s} failed in scenario '{name}'. Read {rel}/log.txt for the error.")


def summarise(name):
    out = ROOT / BASE_OUT / name
    lr = pd.read_csv(out / "learner_results.csv")
    rr = pd.read_pickle(out / "routing_results.pkl")
    cc = pd.read_csv(out / "candidate_counts.csv") if (out / "candidate_counts.csv").exists() else None
    full = lr[(lr.train_frac == lr.train_frac.max()) & (lr.noise == "medium")]
    med = rr[(rr.noise == "") | (rr.noise == "medium")]
    sc = med[med.scored]
    g = lambda m, c: float(sc[sc.method == m][c].mean())
    row = dict(scenario=name, pairs=int(pd.read_csv(out / "od_pairs.csv").shape[0]),
               acc_learned=full.acc_learned.mean(), acc_oracle=full.acc_oracle.mean(), acc_generic=full.acc_generic.mean(),
               weight_mae=full.weight_mae.mean(),
               regret_learned=g("learned", "regret"), regret_generic=g("generic", "regret"),
               regret_shortest_bans=g("shortest_bans", "regret"), regret_shortest_nobans=g("shortest_nobans", "regret"),
               distance_learned=g("learned", "distance_m"), distance_shortest=g("shortest_nobans", "distance_m"),
               nobans_violation_pct=100 * float(sc[sc.method == "shortest_nobans"].violated.mean()))
    if cc is not None:
        row["mean_routes"] = float(cc["n_routes"].mean())
    return row


def main():
    redo = "--redo" in sys.argv
    for name, over in SCENARIOS.items():
        done = (ROOT / BASE_OUT / name / "routing_results.csv").exists()
        if done and not redo:
            print(f"{name}: already finished, skipped")
            continue
        print(f"{name}: {over or 'baseline settings'}", flush=True)
        run(name, over)
    rows = [summarise(n) for n in SCENARIOS if (ROOT / BASE_OUT / n / "routing_results.csv").exists()]
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / BASE_OUT / "summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.3f}".format):
        print("\nOne row per scenario (noise = medium; learner with all training data; test journeys only):")
        print(df.to_string(index=False))
    print("\nsaved outputs/sensitivity/summary.csv")


if __name__ == "__main__":
    main()
