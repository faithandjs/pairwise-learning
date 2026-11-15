"""Shared helpers for steps 12 and 13: routing graphs, route scoring, and the inputs written by step 11."""
import networkx as nx
import numpy as np
import pandas as pd

import config
from common import ban_key, load_segments, split_journeys
from profiles import PROFILES, ban_reasons, banned_mask

FEATURES = ["distance", "gradient", "steps", "kerb"]
RULES = ["steps", "raised_kerb", "steep", "unknown_gradient"]
GENERIC = np.full(4, 0.25)                 # Note-> generic baseline: equal weights on the four features
EPS = 1e-6                                 # tiny share of distance added to every cost so ties go to the shorter route
SPLIT = getattr(config, "SPLIT_FRACTIONS", (0.6, 0.2, 0.2))
SUMMARY_NOISE = getattr(config, "SUMMARY_NOISE", "medium")   # Note-> which noise level the main tables show


class Segs:
    """The segment table as numpy arrays. Row numbers here are the same as row numbers in segments.gpkg."""

    def __init__(self):
        seg = load_segments()
        self.seg = seg
        self.u, self.v = seg["u"].to_numpy(), seg["v"].to_numpy()
        self.length = seg["length_m"].to_numpy(float)
        self.steep = seg["steep_len_m"].fillna(0).to_numpy(float)
        self.steps = seg["steps_length_m"].fillna(0).to_numpy(float)
        self.kerb = seg["kerb_units"].fillna(0).to_numpy(float)
        self.grade = seg["grade_b"].to_numpy(float)                 # fraction; NaN = unknown
        self.unknown = (seg["gradient_unknown"].astype(bool).to_numpy() if "gradient_unknown" in seg
                        else np.isnan(self.grade))
        self.banned = {pid: banned_mask(seg, p["bans"]).to_numpy() for pid, p in PROFILES.items()}
        self.reasons = {pid: {k: v.to_numpy() for k, v in ban_reasons(seg, p["bans"]).items()} for pid, p in PROFILES.items()}


def build_routing_graphs(S):
    """One graph per distinct ban set. Each edge remembers which segment row it came from (attribute 'row')."""
    graphs = {}
    for p in PROFILES.values():
        key = ban_key(p["bans"])
        if key in graphs:
            continue
        mask = banned_mask(S.seg, p["bans"]).to_numpy()
        G = nx.Graph()
        for i in np.where(~mask)[0]:
            u, v = int(S.u[i]), int(S.v[i])
            if u == v:
                continue
            if G.has_edge(u, v) and G[u][v]["length"] <= S.length[i]:
                continue                                           # keep the shortest of any parallel segments
            G.add_edge(u, v, length=S.length[i], steep=S.steep[i], steps=S.steps[i], kerb=S.kerb[i], row=int(i))
        edges = list(G.edges())
        G.graph["e"] = edges
        for name, col in (("L", "length"), ("St", "steep"), ("Sp", "steps"), ("Kb", "kerb")):
            G.graph[name] = np.array([G[a][b][col] for a, b in edges])
        graphs[key] = G
    return graphs


def set_cost(G, w, ref):
    """Edge cost = sum of weight x (feature / reference value), plus a tiny distance term."""
    c = (w[0] * G.graph["L"] / ref[0] + w[1] * G.graph["St"] / ref[1] + w[2] * G.graph["Sp"] / ref[2]
         + w[3] * G.graph["Kb"] / ref[3] + EPS * G.graph["L"])
    nx.set_edge_attributes(G, dict(zip(G.graph["e"], c)), "cost")


def path_rows(G, path):
    return [G[a][b]["row"] for a, b in zip(path[:-1], path[1:])]


def score_route(S, rows, pid, w_true, ref):
    """Measure a route (a list of segment rows) for one profile."""
    rows = np.asarray(rows, dtype=int)
    L = S.length[rows]
    f = np.array([L.sum(), S.steep[rows].sum(), S.steps[rows].sum(), S.kerb[rows].sum()])
    x = np.minimum(f / ref, 1.0)
    g = S.grade[rows]
    ban = S.banned[pid][rows]
    by_rule = {"n_viol_" + k: (int(S.reasons[pid][k][rows].sum()) if k in S.reasons[pid] else np.nan) for k in RULES}
    return dict(distance_m=f[0], steep_m=f[1], steps_m=f[2], kerb_units=f[3],
                max_grade_pct=float(np.nanmax(g) * 100) if (~np.isnan(g)).any() else np.nan,
                climb_m=float(np.nansum(g * L)),                  # total up-and-down in metres (sum of gradient x length)
                unknown_grad_m=float(L[S.unknown[rows]].sum()),
                n_banned_segments=int(ban.sum()), violated=bool(ban.any()), **by_rule,
                utility=float(-(w_true @ x)))


def profile_weights(pid):
    return np.array([PROFILES[pid]["weights"].get(f, 0.0) for f in FEATURES])


def load_learner_inputs():
    """Reference values per repeat, and the learned weights per (repeat, profile, noise level), from step 11."""
    res = pd.read_csv(config.OUTPUTS / "learner_results.csv")
    refs = {int(s): g[["ref_" + f for f in FEATURES]].iloc[0].to_numpy(float) for s, g in res.groupby("seed")}
    wts = pd.read_csv(config.OUTPUTS / "learner_weights.csv")
    learned = {}
    for (s, pid, noise), g in wts.groupby(["seed", "profile", "noise"]):
        vec = np.zeros(4)
        for f, wv in zip(g["feature"], g["learned_weight"]):
            vec[FEATURES.index(f)] = wv
        if not np.isnan(vec).any() and vec.sum() > 0:
            learned[(int(s), int(pid), noise)] = vec
    return refs, learned


def add_regret(df):
    """Regret = utility of the oracle route minus utility of this method's route, for the same repeat, profile and journey."""
    orc = df[df.method == "oracle"].set_index(["seed", "profile", "od_id"])["utility"]
    key = list(zip(df.seed, df.profile, df.od_id))
    df["regret"] = orc.reindex(key).to_numpy() - df["utility"].to_numpy()
    df["violated"] = df["violated"].astype(float)
    return df


def print_summary(df):
    order = ["oracle", "learned", "generic", "shortest_bans", "shortest_nobans", "ors"]
    d = df[(df.noise == "") | (df.noise == SUMMARY_NOISE)]
    rows = []
    for m in order:
        g = d[d.method == m]
        if not len(g):
            continue
        sc = g[g.scored]
        rows.append(dict(method=m, routes=len(g), found_pct=100 * g.found.mean(),
                         violation_pct=100 * sc.violated.mean() if len(sc) else np.nan,
                         distance_m=sc.distance_m.mean(), steep_m=sc.steep_m.mean(), steps_m=sc.steps_m.mean(),
                         kerb_units=sc.kerb_units.mean(), max_grade_pct=sc.max_grade_pct.mean(),
                         climb_m=sc.climb_m.mean(), unknown_grad_m=sc.unknown_grad_m.mean(),
                         regret=sc.regret.mean(), ms_per_route=1000 * g.seconds.mean()))
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.3f}".format):
        print(f"\nAll profiles, all repeats, test journeys only (learned shown at noise = {SUMMARY_NOISE}):")
        print(pd.DataFrame(rows).to_string(index=False))
        print("\nMean regret by profile (lower is better; 0 = the oracle's route):")
        wide = d[d.scored].pivot_table(index="profile", columns="method", values="regret", aggfunc="mean")
        print(wide[[m for m in order if m in wide.columns]].to_string())
        print("\nShare of routes that break the profile's own bans, by profile (%):")
        v = d[d.found].pivot_table(index="profile", columns="method", values="violated", aggfunc="mean") * 100
        print(v[[m for m in order if m in v.columns]].round(1).to_string())
        for m in ("shortest_nobans", "ors"):
            g = d[(d.method == m) & d.scored]
            if not len(g) or "n_viol_steps" not in g:
                continue
            t = pd.DataFrame({k: g.groupby("profile")["n_viol_" + k].apply(lambda s: 100 * (s.dropna() > 0).mean() if s.notna().any() else np.nan)
                              for k in RULES})
            print(f"\n{m}: share of routes breaking each ban rule, by profile (%; blank = the profile has no such rule):")
            print(t.round(1).to_string(na_rep=""))
        L = df[df.method == "learned"]
        if len(L):
            print("\nLearned router, mean regret by noise level:")
            print(L[L.scored].groupby("noise")["regret"].mean().to_string())
