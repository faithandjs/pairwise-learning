"""Step 8b: add connected-piece IDs to od_pairs.csv and summarise where the pairs sit.

Run:  python step8b_od_components.py        (after step8; safe to run again, it does not change the pairs)
Reads:  outputs/od_pairs.csv, data/interim/segments.gpkg
Writes: outputs/od_pairs.csv (extra columns o_comp_<ban set> and d_comp_<ban set>), outputs/od_component_summary.csv

A 'piece' (connected component) is a part of the network where you can get from any point to any other without using
a banned segment. Piece 0 is the longest piece for that ban set, 1 the next longest, and so on (same numbers as
describe_split.py). Each distinct ban set has its own pieces. A profile can only make a journey if the start and end
are in the same piece of its own network.
"""
import pandas as pd

import config
from common import ban_key, build_graphs, component_ids, load_segments
from profiles import PROFILES

# Note-> a piece holding less than this share of the network counts as "small"
SMALL_SHARE = 0.05


def main():
    od = pd.read_csv(config.OUTPUTS / "od_pairs.csv")
    od = od[[c for c in od.columns if not c.startswith(
        ("o_comp_", "d_comp_"))]]
    graphs = build_graphs(load_segments())

    rows = []
    for key, G in graphs.items():
        comp, lengths = component_ids(G)
        total = sum(lengths)
        profs = ", ".join(str(k) for k, p in PROFILES.items()
                          if ban_key(p["bans"]) == key)
        od[f"o_comp_{key}"] = od["origin"].map(comp)
        od[f"d_comp_{key}"] = od["dest"].map(comp)
        o, d = od[f"o_comp_{key}"], od[f"d_comp_{key}"]
        same = (o == d) & o.notna()

        def small(c): return c.map(lambda i: pd.notna(
            i) and lengths[int(i)] / total < SMALL_SHARE)
        rows.append(dict(ban_set=key, profiles=profs, n_pairs=len(od), pairs_same_piece=int(same.sum()),
                         pairs_crossing_pieces=int((~same).sum()),
                         pairs_in_small_pieces=int((same & small(o)).sum()),
                         pairs_in_largest_piece=int((same & (o == 0)).sum())))
        print(f"\nban set '{key}' (profiles {profs}): {len(lengths)} pieces; "
              f"largest {lengths[0]/total:.1%} of unbanned length")
        counts = o[same].value_counts().sort_index()
        for c, n in counts.items():
            print(
                f"   piece {int(c)}: {n} pairs (piece holds {lengths[int(c)]/1000:.2f} km, {lengths[int(c)]/total:.1%})")

    out = pd.DataFrame(rows)
    print("\nSummary (small piece = under {:.0%} of the network):".format(
        SMALL_SHARE))
    print(out.to_string(index=False))
    od.to_csv(config.OUTPUTS / "od_pairs.csv", index=False)
    out.to_csv(config.OUTPUTS / "od_component_summary.csv", index=False)
    print("\nsaved outputs/od_pairs.csv (with comp columns) and outputs/od_component_summary.csv")


if __name__ == "__main__":
    main()
