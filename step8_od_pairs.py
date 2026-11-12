"""Step 8: choose the start/end (origin-destination) pairs that EVERY profile can reach.

Run:  python step8_od_pairs.py
Reads:  data/interim/segments.gpkg, data/interim/nodes.gpkg
Writes: outputs/od_pairs.csv

A pair is kept only if, for every profile's network (after its bans), the start and end are in the same
connected piece, so all eight profiles can make the same journeys. Pairs are at least MIN_SEP_M apart in a
straight line (no trivial journeys) and no node is used twice.
"""
import random

import geopandas as gpd
import networkx as nx
import pandas as pd

import config
from common import build_graphs, load_segments

# Note-> optional: N_OD_PAIRS = 80 in config.py
N_PAIRS = getattr(config, "N_OD_PAIRS", 40)
MIN_SEP_M = getattr(config, "MIN_OD_SEPARATION_M", 500)
SEED = getattr(config, "RANDOM_SEED", 42)
WALKY = {"footway", "pedestrian", "path", "residential", "service", "unclassified", "living_street", "track",
         "tertiary"}   # start/end must touch one of these, so nobody starts in the middle of a trunk road


def main():
    seg = load_segments()
    nodes = gpd.read_file(config.INTERIM / "nodes.gpkg")
    xy = dict(zip(nodes["id"], zip(nodes.geometry.x, nodes.geometry.y)))
    ll = nodes.to_crs(4326)
    latlon = dict(zip(ll["id"], zip(ll.geometry.y, ll.geometry.x)))

    graphs = build_graphs(seg)
    comp = {}
    for key, G in graphs.items():
        d = {}
        for i, c in enumerate(nx.connected_components(G)):
            for n in c:
                d[n] = i
        comp[key] = d

    types = {}
    for r in seg.itertuples():
        types.setdefault(r.u, set()).add(r.highway)
        types.setdefault(r.v, set()).add(r.highway)
    eligible = sorted(n for n in xy if all(
        n in c for c in comp.values()) and types.get(n, set()) & WALKY)
    print(f"{len(graphs)} different bans sets; {len(eligible):,} of {len(xy):,} nodes are present in every "
          f"profile's network and touch a walkable street")
    if len(eligible) < 2:
        raise SystemExit("Too few shared nodes. Check steps 4-6.")

    rng = random.Random(SEED)
    used, pairs, tries = set(), [], 0
    while len(pairs) < N_PAIRS and tries < 500000:
        tries += 1
        a, b = rng.sample(eligible, 2)
        if a in used or b in used:
            continue
        d = ((xy[a][0] - xy[b][0]) ** 2 + (xy[a][1] - xy[b][1]) ** 2) ** 0.5
        if d < MIN_SEP_M or any(c[a] != c[b] for c in comp.values()):
            continue
        used.update((a, b))
        pairs.append((a, b, d))

    rows = []
    for k, (a, b, d) in enumerate(pairs):
        rows.append(dict(od_id=k, origin=a, dest=b, straight_m=round(d, 1),
                         shortest_m=round(nx.dijkstra_path_length(
                             graphs["none"], a, b, weight="length"), 1),
                         origin_lat=round(latlon[a][0], 5), origin_lon=round(latlon[a][1], 5),
                         dest_lat=round(latlon[b][0], 5), dest_lon=round(latlon[b][1], 5)))
    out = pd.DataFrame(rows)
    config.OUTPUTS.mkdir(exist_ok=True)
    out.to_csv(config.OUTPUTS / "od_pairs.csv", index=False)
    print(
        f"chose {len(out)} pairs (wanted {N_PAIRS}), seed {SEED}, at least {MIN_SEP_M} m apart")
    if len(out) < N_PAIRS:
        print("WARNING: fewer pairs than wanted: the bans leave too little shared network. Tell me.")
    if len(out):
        print(
            f"straight-line distance (m): min {out.straight_m.min():.0f}, median {out.straight_m.median():.0f}, max {out.straight_m.max():.0f}")
        print(
            f"shortest route (m):         min {out.shortest_m.min():.0f}, median {out.shortest_m.median():.0f}, max {out.shortest_m.max():.0f}")
    print("saved outputs/od_pairs.csv (open the lat/lon columns on a map if you want to see them)")


if __name__ == "__main__":
    main()
