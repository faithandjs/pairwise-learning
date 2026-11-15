"""Step 12: route over the WHOLE network with the learned weights, and with the in-house baselines.

Run:  python step12_routing.py        (after step 11)
Reads:  data/interim/segments.gpkg, outputs/od_pairs.csv, outputs/candidates.csv,
        outputs/learner_results.csv, outputs/learner_weights.csv
Writes: outputs/routing_results.csv  (one row per repeat x profile x method x test journey)

For each repeat (seed) and each test journey, every method finds the cheapest route (Dijkstra) where
    cost of a segment = sum over the four features of  weight x (feature / reference value)
Methods:
    oracle           the true weights of the profile, with the profile's bans
    learned          the weights learned in step 11 (one set per noise level), with the profile's bans
    generic          equal weights (.25 each), with the profile's bans
    shortest_bans    shortest distance, with the profile's bans
    shortest_nobans  shortest distance, ignoring all bans (plain shortest-path routing)
The ORS wheelchair route is added in step 13.

Every route is then measured the same way: distance, steep metres, steps, kerb units, maximum gradient, total climb
(sum of gradient x length), metres of unknown gradient, whether it uses any segment the profile bans (violation),
and its utility under the profile's TRUE weights. Regret = oracle utility minus this route's utility.
Note: the routing cost adds up rescaled features without the cap at 1 that is used when a whole route is scored,
so on rare long routes the oracle's route can score marginally below another (tiny negative regret).
"""
import time

import networkx as nx
import numpy as np
import pandas as pd

import config
from common import ban_key, split_journeys
from profiles import PROFILES
from routing_common import (GENERIC, Segs, add_regret, build_routing_graphs, load_learner_inputs, path_rows,
                            print_summary, profile_weights, score_route, set_cost, SPLIT)


def route(G, a, b):
    t = time.perf_counter()
    try:
        path = nx.shortest_path(G, a, b, weight="cost")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        path = None
    return path, time.perf_counter() - t


def main():
    S = Segs()
    graphs = build_routing_graphs(S)
    od = pd.read_csv(config.OUTPUTS / "od_pairs.csv").set_index("od_id")
    all_ods = np.array(sorted(pd.read_csv(config.OUTPUTS / "candidates.csv")["od_id"].unique()))
    refs, learned = load_learner_inputs()
    noises = list(getattr(config, "CHOICE_NOISE_LEVELS", {"low": 0.02, "medium": 0.05, "high": 0.10}))
    rows = []

    for seed, ref in refs.items():
        test = split_journeys(all_ods, seed, SPLIT)[2]
        for pid, p in PROFILES.items():
            G, Gn = graphs[ban_key(p["bans"])], graphs["none"]
            w_true = profile_weights(pid)
            methods = [("oracle", "", G, w_true), ("generic", "", G, GENERIC),
                       ("shortest_bans", "", G, np.array([1.0, 0, 0, 0])), ("shortest_nobans", "", Gn, np.array([1.0, 0, 0, 0]))]
            for nz in noises:
                if (seed, pid, nz) in learned:
                    methods.append(("learned", nz, G, learned[(seed, pid, nz)]))
            for name, nz, graph, w in methods:
                set_cost(graph, w, ref)
                for o in test:
                    a, b = int(od.at[o, "origin"]), int(od.at[o, "dest"])
                    path, secs = route(graph, a, b)
                    row = dict(seed=seed, profile=pid, noise=nz, method=name, od_id=int(o), found=path is not None,
                               scored=path is not None, seconds=secs)
                    if path is not None and len(path) > 1:
                        row.update(score_route(S, path_rows(graph, path), pid, w_true, ref))
                    else:
                        row["scored"] = False
                    rows.append(row)

    df = add_regret(pd.DataFrame(rows))
    df.to_csv(config.OUTPUTS / "routing_results.csv", index=False)
    df.to_pickle(config.OUTPUTS / "routing_results.pkl")      # Note-> step 13 reads this copy so nothing is lost in the csv
    bad = df[(df.method.isin(["oracle", "learned", "generic", "shortest_bans"])) & (df.violated == True)]  # noqa: E712
    print(f"{len(df):,} routes saved to outputs/routing_results.csv. "
          f"Check: routes that use a banned segment although the method respects bans: {len(bad)} (should be 0)")
    print_summary(df)


if __name__ == "__main__":
    main()
