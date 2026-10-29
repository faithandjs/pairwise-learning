"""Step 3: audit how much accessibility information OpenStreetMap holds for the study area.

Run:  python step3_audit_osm.py
Reads:  data/interim/edges.gpkg, the OSM extract
Writes: outputs/audit_summary.csv, data/interim/node_tags.csv
"""
import collections

import geopandas as gpd
import osmium
import pandas as pd

import config


class NodeScan(osmium.SimpleHandler):
    """Collect crossing / kerb / tactile-paving tags from nodes inside the study area."""

    def __init__(self):
        super().__init__()
        self.rows = []
        self.nodes_in_box = 0

    def node(self, n):
        if not n.location.valid():
            return
        lon, lat = n.location.lon, n.location.lat
        if not (config.WEST <= lon <= config.EAST and config.SOUTH <= lat <= config.NORTH):
            return
        self.nodes_in_box += 1
        t = n.tags
        keys = ("kerb", "crossing", "tactile_paving", "barrier")
        if t.get("highway") in ("crossing", "traffic_signals") or any(k in t for k in keys):
            self.rows.append({
                "node_id": n.id,
                "highway": t.get("highway"),
                "crossing": t.get("crossing"),
                "kerb": t.get("kerb"),
                "barrier": t.get("barrier"),
                "tactile_paving": t.get("tactile_paving"),
            })


def pct(a, b):
    return f"{a / b:.1%}" if b else "n/a"


def main():
    edges = gpd.read_file(config.INTERIM / "edges.gpkg")
    total_len = edges.length_m.sum()
    rows = []  # (metric, value)

    def add(metric, value):
        rows.append((metric, value))

    add("study-area segments", len(edges))
    add("network length (km)", round(total_len / 1000, 2))

    # ---- steps ----
    steps = edges[edges["highway"] == "steps"]
    add("steps: segments", len(steps))
    add("steps: ways (distinct flights)", steps["id"].nunique())
    add("steps: length (m)", round(steps.length_m.sum(), 1))
    if "step_count" in edges:
        add("steps: ways with step_count tag", f"{steps.groupby('id')['step_count'].apply(lambda s: s.notna().any()).sum()} of {steps['id'].nunique()}")

    # ---- incline / surface / wheelchair tags (share of network length that carries the tag) ----
    for col in ["incline", "surface", "wheelchair", "smoothness"]:
        if col in edges:
            has = edges[col].notna()
            add(f"{col} tag: share of length", pct(edges.loc[has, "length_m"].sum(), total_len))
        else:
            add(f"{col} tag: share of length", "column absent (no such tags in area)")
    if "wheelchair" in edges:
        add("wheelchair tag values", dict(edges["wheelchair"].value_counts()))

    # ---- crossings drawn as separate ways ----
    is_cross = (edges.get("footway") == "crossing") if "footway" in edges else pd.Series(False, index=edges.index)
    cross_ways = edges[is_cross]["id"].nunique()
    add("crossing ways (footway=crossing)", cross_ways)

    # ---- node-level tags: crossings and kerbs (kerb info lives on nodes) ----
    scan = NodeScan()
    scan.apply_file(str(config.OSM_PBF))
    nt = pd.DataFrame(scan.rows)
    nt.to_csv(config.INTERIM / "node_tags.csv", index=False)
    add("nodes scanned inside bounding box", scan.nodes_in_box)
    if len(nt):
        kerb_nodes = nt["kerb"].notna().sum()
        add("nodes with a kerb tag", int(kerb_nodes))
        add("kerb tag values", dict(nt["kerb"].value_counts()))
        add("crossing nodes (highway=crossing)", int((nt["highway"] == "crossing").sum()))
        add("crossing type values", dict(nt["crossing"].value_counts()))
        add("nodes with tactile_paving tag", int(nt["tactile_paving"].notna().sum()))
    else:
        add("nodes with a kerb tag", 0)

    out = pd.DataFrame(rows, columns=["metric", "value"])
    out.to_csv(config.OUTPUTS / "audit_summary.csv", index=False)
    print(out.to_string(index=False))
    print("\nsaved outputs/audit_summary.csv and data/interim/node_tags.csv")


if __name__ == "__main__":
    main()
