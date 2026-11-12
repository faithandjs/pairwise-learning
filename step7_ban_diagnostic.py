"""Step 7: why do the gradient bans cut the network? (follows step 6)

Run:  python step7_ban_diagnostic.py             # looks at the 1-in-12 gradient ban
      python step7_ban_diagnostic.py grade_20    # looks at the 1-in-20 gradient ban
      # Note-> other options: steps, raised_kerb

Reads:  data/interim/segments.gpkg, data/interim/edges_gradient.gpkg, data/interim/nodes.gpkg
Writes: outputs/ban_diagnostic_<ban>.csv and .geojson (open the .geojson at geojson.io to see them on a map)

Part A  which single ban does the damage?
Part B  which banned segments are the only link to a cut-off part of the network, and what are they?
Part C  what-if: how would the connection change if the gradient ban were defined differently?
"""
import sys

import geopandas as gpd
import networkx as nx
import pandas as pd

import config
from profiles import banned_mask
from step6_feasibility import (MIN_SEP_M, N_PAIRS, SEED, build_graph, component_length, evaluate,
                               main_component, pick_pairs)

MIN_PIECE_M = 50   # Note-> cut-off pieces shorter than this are left out of the detail list


def load():
    seg_g = gpd.read_file(config.INTERIM / "segments.gpkg")
    edges = gpd.read_file(config.INTERIM / "edges_gradient.gpkg")
    assert len(seg_g) == len(
        edges), "segments.gpkg and edges_gradient.gpkg differ: re-run steps 4 and 5"
    for c in ("bridge", "tunnel", "layer"):
        seg_g[c] = edges[c].to_numpy() if c in edges else None
    cen = seg_g.geometry.centroid.to_crs(4326)
    seg_g["lat"], seg_g["lon"] = cen.y.round(
        5).to_numpy(), cen.x.round(5).to_numpy()
    return seg_g


def run_set(seg, mask, label, pairs, base_len, full_main_len, total_len):
    G = build_graph(seg[~mask])
    return evaluate(G, pairs, base_len, full_main_len, label, int(mask.sum()), len(seg),
                    float(seg.loc[mask, "length_m"].sum()), total_len)


def critical_links(seg, mask):
    """Banned segments that join the main network to a piece the ban cuts off."""
    G = build_graph(seg[~mask])
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    cid, clen = {}, {}
    for i, comp in enumerate(comps):
        clen[i] = 0.0
        for n in comp:
            cid[n] = i
    for u, v, d in G.edges(data=True):
        clen[cid[u]] += d["length"]
    rows = []
    for r in seg[mask].itertuples():
        cu, cv = cid.get(r.u), cid.get(r.v)
        if cu is None or cv is None or cu == cv or (cu != 0 and cv != 0):
            continue
        lost = cv if cu == 0 else cu
        rows.append((r.Index, lost, clen[lost]))
    links = pd.DataFrame(
        rows, columns=["row", "piece", "lost_piece_m"]).set_index("row")
    if len(links):
        links["n_links"] = links.groupby("piece")["piece"].transform("size")
    return links, (clen[0] if clen else 0.0)


def report_links(seg_g, links, ban):
    det = seg_g.loc[links.index].join(links)
    det["grade_b_pct"] = (det["grade_b"] * 100).round(1)
    det["grade_a_pct"] = (det["grade_a"] * 100).round(1)
    det["osm_link"] = "https://www.openstreetmap.org/way/" + \
        det["id"].astype(str)
    det = det.sort_values(["lost_piece_m", "piece"], ascending=[False, True])
    big = det[det["lost_piece_m"] >= MIN_PIECE_M]
    print(f"{det['piece'].nunique()} cut-off pieces are linked by {len(det)} banned segments; "
          f"{big['piece'].nunique()} pieces are {MIN_PIECE_M} m or longer. Biggest first:\n")
    cols = ["lost_piece_m", "n_links", "id", "highway", "length_m", "grade_b_pct", "grade_a_pct",
            "grade_reliable", "layer", "lat", "lon"]
    shown = big[cols].head(25).round(
        {"lost_piece_m": 1, "length_m": 1, "grade_b_pct": 1, "grade_a_pct": 1})
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(shown.to_string(index=False))
    print("\nsummary of ALL linking segments:")
    print("  by street type:", {k: int(v)
          for k, v in det["highway"].value_counts().items()})
    print(
        f"  shorter than 10 m:                       {100 * (det['length_m'] < 10).mean():.0f}%")
    print(
        f"  gradient measured over under 10 m:       {100 * (~det['grade_reliable'].astype(bool)).mean():.0f}%")
    print(f"  method A (end to end) says gentler than 1 in 20: {100 * (det['grade_a'] < config.STEEP).mean():.0f}%"
          f"   (method A missing: {int(det['grade_a'].isna().sum())})")
    print(f"  median gradient (method B): {det['grade_b_pct'].median():.1f}%")
    out_cols = ["id", "highway", "length_m", "grade_a_pct", "grade_b_pct", "grade_reliable", "gradient_status",
                "layer", "bridge", "tunnel", "lost_piece_m", "n_links", "lat", "lon", "osm_link"]
    config.OUTPUTS.mkdir(exist_ok=True)
    det[out_cols].to_csv(
        config.OUTPUTS / f"ban_diagnostic_{ban}.csv", index=False)
    det[out_cols[:-1] + ["geometry"]].to_crs(4326).to_file(
        config.OUTPUTS / f"ban_diagnostic_{ban}.geojson", driver="GeoJSON")
    print(f"\nsaved outputs/ban_diagnostic_{ban}.csv and .geojson")


def main():
    # Note-> or edit the default here
    ban = sys.argv[1] if len(sys.argv) > 1 else "grade_12"
    if ban not in ("steps", "raised_kerb", "grade_12", "grade_20"):
        raise SystemExit(
            "ban must be one of: steps, raised_kerb, grade_12, grade_20")

    seg_g = load()
    seg = pd.DataFrame(seg_g.drop(columns="geometry"))
    nodes = gpd.read_file(config.INTERIM / "nodes.gpkg")
    nodes_xy = dict(zip(nodes["id"], zip(nodes.geometry.x, nodes.geometry.y)))
    G_full = build_graph(seg)
    full_comp = main_component(G_full)
    full_main_len = component_length(G_full, full_comp)
    total_len = seg["length_m"].sum()
    pairs = pick_pairs(
        nodes_xy, [n for n in full_comp if n in nodes_xy], N_PAIRS, MIN_SEP_M, SEED)
    base_len = [nx.dijkstra_path_length(
        G_full, a, b, weight="length") for a, b in pairs]
    args = (pairs, base_len, full_main_len, total_len)

    # ---------------- Part A
    print("PART A: each ban on its own (compare with step 6)\n")
    singles = {"steps only": ["steps"], "raised kerbs only": ["raised_kerb"],
               "steeper than 1 in 12 only": ["grade_12"], "steeper than 1 in 20 only": ["grade_20"],
               "steps + raised kerbs": ["steps", "raised_kerb"]}
    rows = [run_set(seg, banned_mask(seg, b), name, *args)
            for name, b in singles.items()]
    a = pd.DataFrame(rows)[["profile", "banned_length_pct",
                            "connected_length_pct", "pairs_no_route_pct"]]
    print(a.to_string(index=False))

    # ---------------- Part B
    mask = banned_mask(seg, [ban])
    print(
        f"\nPART B: the {ban} ban on its own: which banned segments cut the network?\n")
    print(
        f"banned: {int(mask.sum())} segments, {seg.loc[mask, 'length_m'].sum():.0f} m")
    links, main_len = critical_links(seg, mask)
    banned_in_main = seg.loc[mask & seg["u"].isin(
        full_comp) & seg["v"].isin(full_comp), "length_m"].sum()
    # length of the pieces cut off (banned segments excluded)
    lost_m = full_main_len - banned_in_main - main_len
    print(
        f"network cut off by this ban: {lost_m:.0f} m ({100 * lost_m / full_main_len:.1f}% of the main network)")
    if links.empty:
        print("no banned segment is the only link to a cut-off piece.")
    else:
        report_links(seg_g, links, ban)

    # ---------------- Part C
    print("\nPART C: what if the 1-in-12 gradient ban were defined differently? (steps are banned in every variant)\n")
    g, rel = seg["grade_b"], seg["grade_reliable"].astype(bool)
    steps, steep = seg["steps_is_steps"].astype(
        bool), seg["grade_b"] > config.VERY_STEEP
    variants = {
        "current: steeper than 1 in 12": steps | steep,
        "only where measured over 10 m or more": steps | (steep & rel),
        "ignore above 15% (treated as unknown)": steps | (steep & (g <= 0.15)),
        "both of the last two": steps | (steep & (g <= 0.15) & rel),
        "no gradient ban (steps only)": steps,
    }
    rows = [run_set(seg, m, name, *args) for name, m in variants.items()]
    c = pd.DataFrame(rows)[["profile", "banned_length_pct", "connected_length_pct", "pairs_no_route_pct",
                            "detour_median_pct"]]
    print(c.to_string(index=False))


if __name__ == "__main__":
    main()
