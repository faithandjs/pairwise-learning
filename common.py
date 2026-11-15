"""Shared helpers for steps 8-10."""
import geopandas as gpd
import pandas as pd

import config
from profiles import PROFILES, banned_mask
from step6_feasibility import build_graph


def ban_key(bans):
    """Profiles with the same bans share one network and one set of candidate routes."""
    return "+".join(sorted(bans)) or "none"


def load_segments():
    return pd.DataFrame(gpd.read_file(config.INTERIM / "segments.gpkg").drop(columns="geometry"))


def build_graphs(seg):
    """One walking graph per distinct set of bans."""
    graphs = {}
    for p in PROFILES.values():
        key = ban_key(p["bans"])
        if key not in graphs:
            graphs[key] = build_graph(seg[~banned_mask(seg, p["bans"])])
    return graphs


def component_ids(G):
    """Connected-piece number for every node: 0 = longest piece, 1 = next longest, and so on (same numbering as describe_split.py)."""
    import networkx as nx
    comps = sorted(nx.connected_components(G),
                   key=lambda c: -sum(d["length"] for *_, d in G.subgraph(c).edges(data=True)))
    lengths = [sum(d["length"] for *_, d in G.subgraph(c).edges(data=True)) for c in comps]
    return {n: i for i, c in enumerate(comps) for n in c}, lengths


def split_journeys(all_ods, seed, split=(0.6, 0.2, 0.2)):
    """Split the origin-destination pairs into train / validation / test by journey (the same way in every step)."""
    import numpy as np
    perm = np.random.default_rng(seed).permutation(np.asarray(all_ods))
    n = len(perm)
    n_tr, n_va = int(round(split[0] * n)), int(round(split[1] * n))
    return perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]
