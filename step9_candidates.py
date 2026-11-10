"""Step 9: candidate routes: 5 to 8 meaningfully different routes per start/end pair, for each set of bans.

Run:  python step9_candidates.py
Reads:  outputs/od_pairs.csv, data/interim/segments.gpkg
Writes: outputs/candidates.csv, outputs/candidate_counts.csv, outputs/candidate_summary.csv

Method (methodology point 4): the first candidate is the shortest allowed route. Each later candidate is found by
making streets used by earlier routes more expensive (cost = length x (1 + PENALTY x times used)) and routing again.
A route is kept only if it shares at most MAX_OVERLAP of its length with every route already kept.
If fewer than MIN_ROUTES are found, we keep what exists and record the shortfall.
"""
import networkx as nx
import pandas as pd

import config
from common import build_graphs, load_segments

MAX_ROUTES = getattr(config, "N_CANDIDATES_MAX", 8)
MIN_ROUTES = getattr(config, "N_CANDIDATES_MIN", 5)
MAX_OVERLAP = getattr(config, "MAX_ROUTE_OVERLAP", 0.6)      # //removethis my default: a route may share up to 60% of its length
PENALTY = getattr(config, "ROUTE_PENALTY", 1.0)
MAX_DETOUR = getattr(config, "MAX_ROUTE_DETOUR", 1.5)        # //removethis NOT in your methodology: drops routes over 1.5x the shortest. Set to None to switch off
MAX_ITER = getattr(config, "MAX_ROUTE_ITERATIONS", 30)


def ekey(u, v):
    return (u, v) if u <= v else (v, u)


def generate(G, a, b):
    used, seen, kept, base_len = {}, set(), [], None
    for _ in range(MAX_ITER):
        def w(u, v, d):
            return d["length"] * (1 + PENALTY * used.get(ekey(u, v), 0))
        path = nx.shortest_path(G, a, b, weight=w)
        edges = [ekey(path[i], path[i + 1]) for i in range(len(path) - 1)]
        for e in edges:
            used[e] = used.get(e, 0) + 1
        if tuple(path) in seen:
            continue
        seen.add(tuple(path))
        length = sum(G[u][v]["length"] for u, v in edges)
        if base_len is None:
            base_len = length
        if MAX_DETOUR and length > MAX_DETOUR * base_len:
            continue
        es = set(edges)
        if all(sum(G[u][v]["length"] for u, v in es & k["edges"]) / length <= MAX_OVERLAP for k in kept):
            kept.append(dict(path=path, edges=es, length=length))
            if len(kept) >= MAX_ROUTES:
                break
    return kept


def main():
    seg = load_segments()
    od = pd.read_csv(config.OUTPUTS / "od_pairs.csv")
    graphs = build_graphs(seg)
    rows, counts, skipped = [], [], 0
    for key, G in graphs.items():
        print(f"bans set '{key}' ...", flush=True)
        for r in od.itertuples():
            try:
                routes = generate(G, r.origin, r.dest)
            except (nx.NodeNotFound, nx.NetworkXNoPath):
                routes, skipped = [], skipped + 1
            counts.append(dict(ban_set=key, od_id=r.od_id, n_routes=len(routes)))
            for k, rt in enumerate(routes):
                e = list(zip(rt["path"][:-1], rt["path"][1:]))
                rows.append(dict(ban_set=key, od_id=r.od_id, route_id=k, distance_m=round(rt["length"], 1),
                                 steep_m=round(sum(G[u][v]["steep"] for u, v in e), 1),
                                 steps_m=round(sum(G[u][v]["steps"] for u, v in e), 1),
                                 kerb_units=round(sum(G[u][v]["kerb"] for u, v in e), 3),
                                 nodes=" ".join(map(str, rt["path"]))))
    if skipped:
        print(f"WARNING: {skipped} pair/profile combinations had no route. Re-run step 8 after any change to steps 4-5.")
    cand, cnt = pd.DataFrame(rows), pd.DataFrame(counts)
    cand.to_csv(config.OUTPUTS / "candidates.csv", index=False)
    cnt.to_csv(config.OUTPUTS / "candidate_counts.csv", index=False)

    summ = []
    for key, g in cand.groupby("ban_set"):
        n_by = cnt[cnt.ban_set == key].set_index("od_id")["n_routes"]
        spread = g.groupby("od_id")[["distance_m", "steep_m", "steps_m", "kerb_units"]].agg(lambda s: s.max() - s.min())
        shortest = g.groupby("od_id")["distance_m"].min()
        longest = g.groupby("od_id")["distance_m"].max()
        summ.append({"ban_set": key, "pairs": len(n_by), "mean_routes": round(n_by.mean(), 1),
                     f"pairs_with_{MIN_ROUTES}+_routes_%": round(100 * (n_by >= MIN_ROUTES).mean(), 0),
                     "pairs_with_1_route_%": round(100 * (n_by <= 1).mean(), 0),
                     "longest_vs_shortest_median_%": round(float((100 * (longest / shortest - 1)).median()), 0),
                     "pairs_steep_differs_%": round(100 * (spread["steep_m"] > 0).mean(), 0),
                     "pairs_steps_differs_%": round(100 * (spread["steps_m"] > 0).mean(), 0),
                     "pairs_kerb_differs_%": round(100 * (spread["kerb_units"] > 0).mean(), 0),
                     "median_steep_range_m": round(float(spread["steep_m"].median()), 1)})
    s = pd.DataFrame(summ)
    s.to_csv(config.OUTPUTS / "candidate_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print()
        print(s.to_string(index=False))
    print("\nsaved outputs/candidates.csv, candidate_counts.csv, candidate_summary.csv")


if __name__ == "__main__":
    main()
