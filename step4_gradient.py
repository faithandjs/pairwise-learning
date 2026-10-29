"""Step 4: attach a gradient to every street segment from the LiDAR DTM.

Run:  python step4_gradient.py
Reads:  data/interim/edges.gpkg, data/interim/nodes.gpkg, the DTM tiles in data/raw/dtm
Writes: data/interim/edges_gradient.gpkg, outputs/gradient_report.csv, outputs/gradient_hist.png

Two gradient methods are computed so you can compare them in the audit:
  A  per segment:  |z_end - z_start| / length                    (noisy on short segments)
  B  windowed:     rise over a WINDOW_M-long stretch of the same OSM way, centred on the segment
"""
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window, from_bounds
from scipy.ndimage import map_coordinates

import config

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------ DTM sampling
class Dtm:
    """Reads only the part of each tile that covers the study area and samples it bilinearly."""

    def __init__(self, folder, patterns, bounds, pad=30.0):
        paths = sorted({p for pat in patterns for p in folder.glob(pat)})
        if not paths:
            raise SystemExit(f"No DTM tiles found in {folder} (looked for {patterns})")
        minx, miny, maxx, maxy = bounds
        self.tiles = []
        for p in paths:
            with rasterio.open(p) as src:
                if src.crs is None:
                    print(f"  {p.name}: no CRS stored, assuming {config.DTM_CRS_IF_MISSING}")
                elif src.crs.to_epsg() != int(config.PROJECTED_CRS.split(":")[1]):
                    raise SystemExit(f"{p.name} is in {src.crs}, not {config.PROJECTED_CRS}. "
                                     f"Reproject it first, e.g.  gdalwarp -t_srs {config.PROJECTED_CRS} in.tif out.tif")
                win = from_bounds(minx - pad, miny - pad, maxx + pad, maxy + pad, src.transform)
                try:
                    win = win.round_offsets().round_lengths().intersection(Window(0, 0, src.width, src.height))
                except Exception:
                    continue  # tile does not overlap the study area
                if win.width < 1 or win.height < 1:
                    continue
                arr = src.read(1, window=win, masked=True).astype("float32").filled(np.nan)
                self.tiles.append((src.window_transform(win), arr, p.name))
        print(f"  {len(self.tiles)} of {len(paths)} tiles overlap the study area")
        if not self.tiles:
            raise SystemExit("None of the DTM tiles overlap the study area.")

    def sample(self, xs, ys):
        out = np.full(len(xs), np.nan)
        for tr, arr, _ in self.tiles:
            h, w = arr.shape
            left, top = tr.c, tr.f
            right, bottom = left + w * tr.a, top + h * tr.e      # tr.e is negative for north-up rasters
            # "inside" uses the tile's outer edge, so points within half a pixel of a tile
            # boundary still get a value (nearest-pixel there) instead of falling between tiles.
            inside = (xs >= left) & (xs < right) & (ys > bottom) & (ys <= top) & np.isnan(out)
            if inside.any():
                cols = (xs[inside] - left) / tr.a - 0.5            # pixel-centre coordinates
                rows = (ys[inside] - top) / tr.e - 0.5
                out[inside] = map_coordinates(arr, [rows, cols], order=1, mode="nearest")
        return out


# ------------------------------------------------------------------ gradient helpers
def order_chain(g):
    """Order the segments of one OSM way end to end. Returns (node_sequence, edge_index_sequence) or None."""
    adj = {}
    for idx, u, v in zip(g.index, g["u"], g["v"]):
        adj.setdefault(u, []).append((v, idx))
        adj.setdefault(v, []).append((u, idx))
    if any(len(a) > 2 for a in adj.values()):
        return None
    ends = [n for n, a in adj.items() if len(a) == 1]
    if len(ends) not in (0, 2):
        return None
    start = ends[0] if ends else next(iter(adj))
    nodes, edge_ids, used, cur = [start], [], set(), start
    while True:
        nxt = [(n, i) for n, i in adj[cur] if i not in used]
        if not nxt:
            break
        n, i = nxt[0]
        used.add(i); edge_ids.append(i); nodes.append(n); cur = n
    if len(edge_ids) != len(g):
        return None
    return nodes, edge_ids


def windowed_grades(edges, z_by_node):
    grade_b = pd.Series(np.nan, index=edges.index)
    window_used = pd.Series(np.nan, index=edges.index)
    fallback = 0
    for _, g in edges.groupby("id"):
        chain = order_chain(g)
        if chain is None:
            fallback += len(g)
            continue
        nodes, eids = chain
        seg_len = edges.loc[eids, "length_m"].to_numpy()
        s = np.concatenate([[0.0], np.cumsum(seg_len)])
        zs = np.array([z_by_node.get(n, np.nan) for n in nodes])
        total = s[-1]
        for i, eid in enumerate(eids):
            mid = (s[i] + s[i + 1]) / 2
            lo, hi = max(0.0, mid - config.WINDOW_M / 2), min(total, mid + config.WINDOW_M / 2)
            if hi - lo <= 0:
                continue
            # skip this segment if any node that the window touches has no elevation
            i0 = max(0, np.searchsorted(s, lo, side="right") - 1)
            i1 = min(len(s) - 1, np.searchsorted(s, hi, side="left"))
            if np.isnan(zs[i0:i1 + 1]).any():
                continue
            grade_b[eid] = abs(np.interp(hi, s, zs) - np.interp(lo, s, zs)) / (hi - lo)
            window_used[eid] = hi - lo
    return grade_b, window_used, fallback


def parse_incline(v):
    """OSM incline tag -> absolute gradient as a fraction. 'up'/'down'/plain numbers are ignored."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    s = str(v).strip().lower().replace(" ", "")
    m = re.fullmatch(r"[-+]?(\d+(?:\.\d+)?)%", s)
    if m:
        return float(m.group(1)) / 100
    m = re.fullmatch(r"[-+]?(\d+(?:\.\d+)?)°", s)
    if m:
        return float(np.tan(np.radians(float(m.group(1)))))
    return np.nan


def wl(mask, lengths):
    """Length-weighted share."""
    return float(lengths[mask].sum() / lengths.sum()) if lengths.sum() else np.nan


def main():
    edges = gpd.read_file(config.INTERIM / "edges.gpkg")
    nodes = gpd.read_file(config.INTERIM / "nodes.gpkg")

    minx, miny, maxx, maxy = nodes.total_bounds
    print("Reading DTM tiles ...")
    dtm = Dtm(config.DTM_DIR, config.DTM_PATTERNS, (minx, miny, maxx, maxy))

    nodes["z"] = dtm.sample(nodes.geometry.x.to_numpy(), nodes.geometry.y.to_numpy())
    z_by_node = dict(zip(nodes["id"], nodes["z"]))
    cover = nodes["z"].notna().mean()
    print(f"nodes with an elevation: {cover:.1%}")

    edges["z_u"] = edges["u"].map(z_by_node)
    edges["z_v"] = edges["v"].map(z_by_node)
    edges["grade_a"] = (edges["z_v"] - edges["z_u"]).abs() / edges["length_m"].clip(lower=0.5)
    edges["grade_b"], edges["window_m"], n_fallback = windowed_grades(edges, z_by_node)
    edges["grade_b_fallback"] = edges["grade_b"].isna() & edges["grade_a"].notna()
    edges["grade_b"] = edges["grade_b"].fillna(edges["grade_a"])
    edges["grade_reliable"] = edges["window_m"].fillna(edges["length_m"]) >= config.MIN_RELIABLE_M

    # Bridges and tunnels: the DTM shows the ground, not the deck, so the gradient there is
    # unreliable. Ignore the gradient on any way tagged bridge or tunnel (not 'no').
    on_struct = pd.Series(False, index=edges.index)
    for col in ("bridge", "tunnel"):
        if col in edges:
            s = edges[col].astype("string").str.lower()
            on_struct |= (s.notna() & ~s.isin(["no", "false", "0", ""])).fillna(False).astype(bool)
    edges["on_structure"] = on_struct
    edges.loc[on_struct, ["grade_a", "grade_b"]] = np.nan
    edges["gradient_status"] = np.where(on_struct, "excluded_bridge_or_tunnel",
                                        np.where(edges["grade_b"].notna(), "measured", "no_elevation"))

    edges["steep_len_b_m"] = np.where(edges["grade_b"] > config.STEEP, edges["length_m"], 0.0)
    edges["very_steep_b"] = edges["grade_b"] > config.VERY_STEEP

    if "incline" in edges:
        edges["incline_abs"] = edges["incline"].map(parse_incline)
    else:
        edges["incline_abs"] = np.nan

    L = edges["length_m"]
    ok = edges["grade_b"].notna()
    rep = {
        "nodes with elevation": f"{cover:.1%}",
        "segments with a gradient": f"{ok.mean():.1%}",
        "bridge/tunnel segments with gradient ignored": int(on_struct.sum()),
        "bridge/tunnel length with gradient ignored (m)": round(float(L[on_struct].sum()), 0),
        "segments STILL steeper than 20% after that (check these)": int((edges["grade_b"] > 0.20).sum()),
        "segments where method B fell back to method A": int(edges["grade_b_fallback"].sum()),
        "ways with an odd shape (no windowed gradient): segments": n_fallback,
        "median segment length (m)": round(L.median(), 1),
        "share of segments shorter than 10 m": f"{(L < 10).mean():.0%}",
        "length share steeper than 1 in 20 (method A)": f"{wl((edges.grade_a > config.STEEP) & ok, L):.1%}",
        "length share steeper than 1 in 20 (method B)": f"{wl((edges.grade_b > config.STEEP) & ok, L):.1%}",
        "length share steeper than 1 in 12 (method B)": f"{wl((edges.grade_b > config.VERY_STEEP) & ok, L):.1%}",
        "segments where A and B disagree on 'steep'": f"{((edges.grade_a > config.STEEP) != (edges.grade_b > config.STEEP))[ok].mean():.1%}",
    }
    for q in (0.5, 0.9, 0.95, 0.99):
        rep[f"gradient percentile {int(q*100)} (method B)"] = f"{edges.loc[ok, 'grade_b'].quantile(q):.2%}"

    inc = edges[edges["incline_abs"].notna() & ok]
    rep["segments with a numeric OSM incline tag"] = len(inc)
    if len(inc) >= 5:
        rep["OSM incline vs method B: correlation"] = round(float(np.corrcoef(inc.incline_abs, inc.grade_b)[0, 1]), 2)
        rep["OSM incline minus method B: mean (percentage points)"] = round(float((inc.incline_abs - inc.grade_b).mean() * 100), 2)
        rep["OSM incline vs method B: mean absolute difference (pp)"] = round(float((inc.incline_abs - inc.grade_b).abs().mean() * 100), 2)

    out = pd.DataFrame({"metric": list(rep), "value": list(rep.values())})
    out.to_csv(config.OUTPUTS / "gradient_report.csv", index=False)
    print(out.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(edges.loc[ok, "grade_b"] * 100, bins=60, weights=L[ok], color="#4477aa")
    ax.axvline(config.STEEP * 100, color="#cc6677", ls="--", label="1 in 20")
    ax.axvline(config.VERY_STEEP * 100, color="#882255", ls="--", label="1 in 12")
    ax.set_xlabel("gradient (%), method B"); ax.set_ylabel("network length (m)"); ax.legend()
    fig.savefig(config.OUTPUTS / "gradient_hist.png", dpi=150, bbox_inches="tight")

    edges.to_file(config.INTERIM / "edges_gradient.gpkg", layer="edges", driver="GPKG")
    print("\nsaved data/interim/edges_gradient.gpkg, outputs/gradient_report.csv, outputs/gradient_hist.png")


if __name__ == "__main__":
    main()
