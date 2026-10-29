"""Step 2: load the walkable street network for the study area from the frozen OSM extract.

Run:  python step2_load_network.py
Writes: data/interim/edges.gpkg, data/interim/nodes.gpkg, outputs/network_map.png
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyrosm import OSM
from shapely.geometry import box

import config

warnings.filterwarnings("ignore")

# Extra way tags to keep if they exist in the extract (missing ones are skipped silently).
EXTRA = ["incline", "step_count", "kerb", "crossing", "tactile_paving", "ramp",
         "wheelchair", "smoothness", "surface", "layer"]
KEEP = ["id", "u", "v", "highway", "footway", "name", "surface", "incline", "step_count",
        "kerb", "crossing", "tactile_paving", "ramp", "wheelchair", "smoothness", "tags",
        "bridge", "tunnel", "layer", "length_m", "geometry"]


def main():
    config.INTERIM.mkdir(parents=True, exist_ok=True)
    config.OUTPUTS.mkdir(parents=True, exist_ok=True)
    if not config.OSM_PBF.exists():
        raise SystemExit(f"OSM file not found: {config.OSM_PBF}\nPut your Geofabrik .osm.pbf there or edit config.py")

    bbox = box(config.WEST, config.SOUTH, config.EAST, config.NORTH)  # west, south, east, north
    osm = OSM(str(config.OSM_PBF), bounding_box=bbox)
    nodes, edges = osm.get_network(network_type="walking", nodes=True, extra_attributes=EXTRA)
    if edges is None or len(edges) == 0:
        raise SystemExit("No walkable edges found: check the bounding box and the PBF file.")

    nodes = nodes.to_crs(config.PROJECTED_CRS)
    edges = edges.to_crs(config.PROJECTED_CRS)
    edges["length_m"] = edges.geometry.length
    edges = edges[[c for c in KEEP if c in edges.columns]].copy()
    nodes = nodes[[c for c in ["id", "geometry"] if c in nodes.columns]].copy()

    edges.to_file(config.INTERIM / "edges.gpkg", layer="edges", driver="GPKG")
    nodes.to_file(config.INTERIM / "nodes.gpkg", layer="nodes", driver="GPKG")

    print(f"nodes: {len(nodes):,}   segments: {len(edges):,}   network length: {edges.length_m.sum()/1000:.1f} km")
    print(f"segment length (m): median {edges.length_m.median():.1f}, "
          f"10th pct {edges.length_m.quantile(.1):.1f}, 90th pct {edges.length_m.quantile(.9):.1f}")
    print(f"segments shorter than 10 m: {(edges.length_m < 10).mean():.0%}")
    print("segments by highway type:")
    print(edges["highway"].value_counts().to_string())

    fig, ax = plt.subplots(figsize=(8, 8))
    edges.plot(ax=ax, linewidth=0.6, color="#333333")
    ax.set_title("Walkable network, study area (check this looks right)")
    ax.set_axis_off()
    fig.savefig(config.OUTPUTS / "network_map.png", dpi=150, bbox_inches="tight")
    print("saved outputs/network_map.png")


if __name__ == "__main__":
    main()
