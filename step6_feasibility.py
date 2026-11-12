"""Step 6: feasibility check for the eight profiles (profile.md, Section 7).

Run:  python step6_feasibility.py
Reads:  data/interim/segments.gpkg, data/interim/nodes.gpkg
Writes: outputs/feasibility.csv

For each profile it answers:
  - how much of the network do the bans remove?
  - how much stays connected to the main network?
  - what share of random start/end pairs have NO route? how much longer are the routes (detour)?
  - on the shortest route for each pair, how often does each feature actually appear?
    (a feature that almost never appears gives the learning model nothing to learn from)
"""
import random

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd

import config
from profiles import PROFILES, banned_mask

# Note-> optional: add N_FEASIBILITY_PAIRS = 500 to config.py
N_PAIRS = getattr(config, "N_FEASIBILITY_PAIRS", 300)
# Note-> start and end must be at least this far apart (straight line)
MIN_SEP_M = getattr(config, "MIN_OD_SEPARATION_M", 500)
SEED = getattr(config, "RANDOM_SEED", 42)


def build_graph(seg):
    G = nx.Graph()
    for r in seg.itertuples():
        if r.u == r.v:
            continue
        if G.has_edge(r.u, r.v) and G[r.u][r.v]["length"] <= r.length_m:
            continue  # keep the shortest of any parallel segments
        G.add_edge(r.u, r.v, length=r.length_m, steps=r.steps_length_m,
                   steep=r.steep_len_m, kerb=r.kerb_units)
    return G


def main_component(G):
    return max(nx.connected_components(G), key=len)


def component_length(G, comp):
    return sum(d["length"] for u, v, d in G.edges(data=True) if u in comp and v in comp)


def pick_pairs(nodes_xy, candidates, n, min_sep, seed):
    rng = random.Random(seed)
    cand = sorted(candidates)
    pairs, tries = [], 0
    while len(pairs) < n and tries < n * 200:
        tries += 1
        a, b = rng.sample(cand, 2)
        (xa, ya), (xb, yb) = nodes_xy[a], nodes_xy[b]
        if ((xa - xb) ** 2 + (ya - yb) ** 2) ** 0.5 >= min_sep:
            pairs.append((a, b))
    return pairs


def evaluate(G, pairs, base_len, full_main_len, label, n_banned_seg, n_seg, banned_len, total_len):
    comp = main_component(G) if G.number_of_nodes() else set()
    rows = dict(profile=label,
                banned_segments_pct=round(100 * n_banned_seg / n_seg, 1),
                banned_length_pct=round(100 * banned_len / total_len, 1),
                connected_length_pct=round(100 * component_length(G, comp) / full_main_len, 1) if comp else 0.0)
    no_route, detours, has_steps, has_steep, has_kerb, steep_m = 0, [], [], [], [], []
    for (a, b), L0 in zip(pairs, base_len):
        if a not in G or b not in G or not nx.has_path(G, a, b):
            no_route += 1
            continue
        path = nx.shortest_path(G, a, b, weight="length")
        edges = [G[path[i]][path[i + 1]] for i in range(len(path) - 1)]
        L = sum(e["length"] for e in edges)
        detours.append(100 * (L / L0 - 1))
        s, g, k = sum(e["steps"] for e in edges), sum(e["steep"]
                                                      for e in edges), sum(e["kerb"] for e in edges)
        has_steps.append(s > 0)
        has_steep.append(g > 0)
        has_kerb.append(k > 0)
        steep_m.append(g)
    n = len(pairs)
    ok = n - no_route
    rows.update(pairs_no_route_pct=round(100 * no_route / n, 1),
                detour_median_pct=round(
                    float(np.median(detours)), 1) if ok else np.nan,
                detour_p90_pct=round(
                    float(np.percentile(detours, 90)), 1) if ok else np.nan,
                routes_with_steps_pct=round(
                    100 * np.mean(has_steps), 1) if ok else np.nan,
                routes_with_steep_pct=round(
                    100 * np.mean(has_steep), 1) if ok else np.nan,
                routes_with_unconfirmed_kerb_pct=round(
                    100 * np.mean(has_kerb), 1) if ok else np.nan,
                median_steep_m=round(float(np.median(steep_m)), 1) if ok else np.nan)
    return rows


def main():
    seg = pd.DataFrame(gpd.read_file(
        config.INTERIM / "segments.gpkg").drop(columns="geometry"))
    nodes = gpd.read_file(config.INTERIM / "nodes.gpkg")
    nodes_xy = dict(zip(nodes["id"], zip(nodes.geometry.x, nodes.geometry.y)))

    G_full = build_graph(seg)
    full_comp = main_component(G_full)
    full_main_len = component_length(G_full, full_comp)
    total_len = seg["length_m"].sum()
    print(f"network: {len(seg):,} segments, {G_full.number_of_nodes():,} nodes; main connected piece holds "
          f"{100 * full_main_len / total_len:.1f}% of the length")
    unknown_grad = seg[seg["gradient_status"] != "measured"]
    print(f"segments with unknown gradient (banned only for profiles with a gradient limit, see UNKNOWN_GRADIENT_BAN): {len(unknown_grad):,} "
          f"({100 * len(unknown_grad) / len(seg):.1f}% of segments, {unknown_grad['length_m'].sum():.0f} m)")

    candidates = [n for n in full_comp if n in nodes_xy]
    pairs = pick_pairs(nodes_xy, candidates, N_PAIRS, MIN_SEP_M, SEED)
    print(f"{len(pairs)} random start/end pairs, at least {MIN_SEP_M} m apart, seed {SEED}")
    base_len = [nx.dijkstra_path_length(
        G_full, a, b, weight="length") for a, b in pairs]

    rows = [evaluate(G_full, pairs, base_len, full_main_len,
                     "0 none (reference)", 0, len(seg), 0.0, total_len)]
    for k, p in PROFILES.items():
        mask = banned_mask(seg, p["bans"])
        G = build_graph(seg[~mask])
        rows.append(evaluate(G, pairs, base_len, full_main_len, f"{k} {p['name']}", int(mask.sum()), len(seg),
                             float(seg.loc[mask, "length_m"].sum()), total_len))
    out = pd.DataFrame(rows)
    config.OUTPUTS.mkdir(exist_ok=True)
    out.to_csv(config.OUTPUTS / "feasibility.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print()
        print(out.to_string(index=False))
    print("\nsaved outputs/feasibility.csv")


if __name__ == "__main__":
    main()
