"""Describe how a set of bans splits the street network into pieces, and which banned segments join the pieces.

Run:  python describe_split.py                      (profile 3's bans: steps + steeper than 1 in 12)
      python describe_split.py grade_20+raised_kerb+steps     (profile 5's bans; any key from the reach_*.png names)
Reads:  data/interim/segments.gpkg
Writes: outputs/split_links_<ban set>.csv  (banned segments that join two different pieces, with coordinates)
"""
import sys

import geopandas as gpd
import networkx as nx
import pandas as pd

import config
from common import ban_key, build_graphs
from profiles import PROFILES, banned_mask


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "grade_12+steps"
    geo = gpd.read_file(config.INTERIM / "segments.gpkg")
    seg = pd.DataFrame(geo.drop(columns="geometry"))
    graphs = build_graphs(seg)
    if key not in graphs:
        raise SystemExit(f"No such ban set. Choose one of: {', '.join(graphs)}")
    G = graphs[key]
    bans = next(p["bans"] for p in PROFILES.values() if ban_key(p["bans"]) == key)
    banned = banned_mask(seg, bans).to_numpy()
    comps = sorted(nx.connected_components(G), key=lambda c: -sum(d["length"] for *_, d in G.subgraph(c).edges(data=True)))
    piece = {n: i for i, c in enumerate(comps) for n in c}
    ll = geo.to_crs(4326)
    total = geo.loc[~banned, "length_m"].sum()

    print(f"ban set '{key}': {len(comps)} connected pieces")
    print(f"{'piece':>5} {'length_km':>10} {'% of unbanned':>14} {'centre lat':>11} {'centre lon':>11}")
    for i, c in enumerate(comps[:6]):
        m = geo["u"].isin(c) & geo["v"].isin(c) & ~banned
        cen = ll[m].geometry.union_all().centroid
        L = geo.loc[m, "length_m"].sum()
        print(f"{i:>5} {L/1000:>10.2f} {L/total:>13.1%} {cen.y:>11.5f} {cen.x:>11.5f}")

    rows = []
    for idx in geo.index[banned]:
        a, b = piece.get(geo.at[idx, "u"]), piece.get(geo.at[idx, "v"])
        if a is not None and b is not None and a != b:
            mid = ll.geometry[idx].interpolate(0.5, normalized=True)
            rows.append(dict(way_id=geo.at[idx, "id"], highway=geo.at[idx, "highway"], length_m=round(geo.at[idx, "length_m"], 1),
                             grade_b_pct=round(100 * geo.at[idx, "grade_b"], 1) if pd.notna(geo.at[idx, "grade_b"]) else None,
                             kerb_status=geo.at[idx, "kerb_status"], joins_piece=a, and_piece=b,
                             lat=round(mid.y, 5), lon=round(mid.x, 5)))
    out = pd.DataFrame(rows).sort_values(["way_id"]) if rows else pd.DataFrame()
    out.to_csv(config.OUTPUTS / f"split_links_{key}.csv", index=False)
    print(f"\nbanned segments that directly join two different pieces: {len(out)}")
    if len(out):
        print(out.to_string(index=False))
    print(f"saved outputs/split_links_{key}.csv")


if __name__ == "__main__":
    main()
