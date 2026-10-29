"""Step 5: build the segment table that every later step reads from.

Run:  python step5_segment_table.py
Reads:  data/interim/edges_gradient.gpkg, data/interim/node_tags.csv
Writes: outputs/segment_table.csv, data/interim/segments.gpkg

One row per street segment, with the four route features as additive per-segment values:
  distance  -> length_m
  gradient  -> steep_len_m        (metres steeper than 1 in 20, method B)
  steps     -> steps_length_m     (metres of steps; steps_is_steps and step_count_tag kept for reference)
  kerb      -> kerb_units         (1 per crossing not confirmed as lowered/flush, spread over its segments)
plus kerb_raised_known (a known raised kerb, for hard constraints) and the kerb_status label.
"""
import geopandas as gpd
import numpy as np
import pandas as pd

import config

GOOD = {"lowered", "flush"}
BAD = {"raised"}


def crossing_status(g, kerb_by_node):
    """Kerb status for one crossing way, judged from the kerb tags on its two end nodes."""
    counts = pd.concat([g["u"], g["v"]]).value_counts()
    ends = list(counts[counts == 1].index)
    if len(ends) != 2:                       # odd shape: treat as unknown
        return "unknown"
    vals = [kerb_by_node.get(n) for n in ends]
    if any(v in BAD for v in vals):
        return "raised"
    if all(v in GOOD for v in vals):
        return "ok"
    if any(isinstance(v, str) and v not in GOOD | BAD for v in vals):
        return "other"                       # a kerb value we did not expect: inspect these
    return "unknown"


def main():
    edges = gpd.read_file(config.INTERIM / "edges_gradient.gpkg")
    nt = pd.read_csv(config.INTERIM / "node_tags.csv", dtype={"kerb": "string"})
    kerb_by_node = dict(zip(nt["node_id"], nt["kerb"])) if len(nt) else {}

    # ---- steps ----
    edges["steps_is_steps"] = edges["highway"] == "steps"
    edges["steps_length_m"] = np.where(edges["steps_is_steps"], edges["length_m"], 0.0)
    edges["step_count_tag"] = edges["step_count"] if "step_count" in edges else np.nan

    # ---- kerbs on crossings ----
    footway = edges["footway"] if "footway" in edges else pd.Series("", index=edges.index)
    edges["is_crossing"] = (footway == "crossing") | (edges["highway"] == "crossing")
    edges["kerb_status"] = "not_crossing"
    edges["kerb_units"] = 0.0
    for _, g in edges[edges["is_crossing"]].groupby("id"):
        status = crossing_status(g, kerb_by_node)
        edges.loc[g.index, "kerb_status"] = status
        edges.loc[g.index, "kerb_units"] = (0.0 if status == "ok" else 1.0) / len(g)
    edges["kerb_raised_known"] = edges["kerb_status"] == "raised"

    # Steps are already counted by the steps feature, so they add nothing to the gradient feature.
    edges.loc[edges["steps_is_steps"], "steep_len_b_m"] = 0.0

    cols = ["id", "u", "v", "highway", "length_m", "grade_a", "grade_b", "grade_reliable",
            "on_structure", "gradient_status", "grade_b_raw",
            "steep_len_b_m", "very_steep_b", "steps_is_steps", "steps_length_m", "step_count_tag",
            "is_crossing", "kerb_status", "kerb_units", "kerb_raised_known", "incline_abs", "geometry"]
    seg = edges[[c for c in cols if c in edges.columns]].rename(columns={"steep_len_b_m": "steep_len_m"})
    seg.to_file(config.INTERIM / "segments.gpkg", layer="segments", driver="GPKG")
    pd.DataFrame(seg.drop(columns="geometry")).to_csv(config.OUTPUTS / "segment_table.csv", index=False)

    n_cross_ways = edges.loc[edges["is_crossing"], "id"].nunique()
    print(f"segments: {len(seg):,}")
    print(f"steps: {int(seg.steps_is_steps.sum())} segments, {seg.steps_length_m.sum():.0f} m")
    print(f"gradient: {seg.steep_len_m.sum():.0f} m of {seg.length_m.sum():.0f} m steeper than 1 in 20 (method B)")
    print(f"crossing ways: {n_cross_ways}")
    if n_cross_ways:
        by_way = edges[edges["is_crossing"]].groupby("id")["kerb_status"].first().value_counts()
        print("crossing kerb status (per crossing way):")
        print(by_way.to_string())
    print("saved outputs/segment_table.csv and data/interim/segments.gpkg")


if __name__ == "__main__":
    main()
