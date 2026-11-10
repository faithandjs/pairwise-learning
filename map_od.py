"""Draw maps of the network pieces each set of bans leaves, with the origin-destination pairs on top.

Run:  python map_od.py
Reads:  data/interim/segments.gpkg, data/interim/nodes.gpkg, outputs/od_pairs.csv
Writes: outputs/reach_<ban set>.png   (one map per distinct ban set)
Colours: grey = banned segment, blue = biggest connected piece, orange = every other piece (cut off),
         green dot = start, red dot = end, thin black line = straight line between them.
"""
import geopandas as gpd
import matplotlib
import networkx as nx
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from common import ban_key, build_graphs
from profiles import PROFILES, banned_mask


def main():
    geo = gpd.read_file(config.INTERIM / "segments.gpkg")
    seg = pd.DataFrame(geo.drop(columns="geometry"))
    nodes = gpd.read_file(config.INTERIM / "nodes.gpkg").set_index("id")
    pairs = pd.read_csv(config.OUTPUTS / "od_pairs.csv")
    graphs = build_graphs(seg)

    for key, G in graphs.items():
        users = [str(k) for k, p in PROFILES.items() if ban_key(p["bans"]) == key]
        bans = next(p["bans"] for p in PROFILES.values() if ban_key(p["bans"]) == key)
        banned = banned_mask(seg, bans).to_numpy()
        main_comp = max(nx.connected_components(G), key=len)
        in_main = (geo["u"].isin(main_comp) & geo["v"].isin(main_comp)).to_numpy()

        fig, ax = plt.subplots(figsize=(9, 9))
        if banned.any():
            geo[banned].plot(ax=ax, color="#bbbbbb", linewidth=1.2)
        if (~banned & ~in_main).any():
            geo[~banned & ~in_main].plot(ax=ax, color="#e67e22", linewidth=1.2)
        geo[~banned & in_main].plot(ax=ax, color="#2a6fb0", linewidth=0.8)
        for r in pairs.itertuples():
            a, b = nodes.loc[r.origin].geometry, nodes.loc[r.dest].geometry
            ax.plot([a.x, b.x], [a.y, b.y], color="black", linewidth=0.6, alpha=0.6)
            ax.scatter(a.x, a.y, c="green", s=22, zorder=5)
            ax.scatter(b.x, b.y, c="red", s=22, zorder=5)
        share = geo.loc[~banned & in_main, "length_m"].sum() / geo.loc[~banned, "length_m"].sum()
        ax.set_title(f"Profiles {', '.join(users)} (bans: {key})\n"
                     f"blue = biggest connected piece ({share:.0%} of unbanned length), orange = cut off, grey = banned")
        ax.set_axis_off()
        out = config.OUTPUTS / f"reach_{key}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("saved", out)


if __name__ == "__main__":
    main()
