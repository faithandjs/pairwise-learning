"""Step 1: check the environment and your input files before running anything else.

Run:  python step1_check_inputs.py
"""
import importlib

import config

print("--- libraries ---")
for name in ["pyrosm", "osmium", "rasterio", "geopandas", "shapely", "pyproj", "scipy", "pandas", "numpy", "matplotlib"]:
    try:
        m = importlib.import_module(name)
        print(f"{name:12s} {getattr(m, '__version__', 'ok')}")
    except Exception as e:  # noqa: BLE001
        print(f"{name:12s} MISSING ({e})")

print("\n--- OSM extract ---")
if config.OSM_PBF.exists():
    print("found:", config.OSM_PBF, f"({config.OSM_PBF.stat().st_size / 1e6:.0f} MB)")
    try:
        import osmium
        with osmium.io.Reader(str(config.OSM_PBF), osmium.osm.osm_entity_bits.NOTHING) as r:
            h = r.header()
            print("extract timestamp in file header:", h.get("osmosis_replication_timestamp") or "not recorded")
    except Exception as e:  # noqa: BLE001
        print("could not read header:", e)
    print("record the Geofabrik download date in your notes (your proposal says 1 Aug 2026)")
else:
    print("NOT FOUND:", config.OSM_PBF)

print("\n--- DTM tiles ---")
paths = sorted({p for pat in config.DTM_PATTERNS for p in config.DTM_DIR.glob(pat)})
if not paths:
    print("NO TILES FOUND in", config.DTM_DIR)
else:
    import rasterio
    from pyproj import Transformer
    from shapely.geometry import box
    from shapely.ops import unary_union

    tr = Transformer.from_crs("EPSG:4326", config.PROJECTED_CRS, always_xy=True)
    xs, ys = zip(*[tr.transform(x, y) for x, y in
                   [(config.WEST, config.SOUTH), (config.EAST, config.SOUTH),
                    (config.EAST, config.NORTH), (config.WEST, config.NORTH)]])
    area = box(min(xs), min(ys), max(xs), max(ys))
    boxes = []
    for p in paths:
        with rasterio.open(p) as s:
            crs = s.crs.to_string() if s.crs else "NONE STORED"
            print(f"{p.name}: crs={crs}, pixel={s.res[0]:.2f} m, size={s.width}x{s.height}, nodata={s.nodata}")
            boxes.append(box(*s.bounds))
    covered = unary_union(boxes).intersection(area).area / area.area
    print(f"\ntiles cover {covered:.1%} of the study area (bounding box in {config.PROJECTED_CRS})")
    if covered < 0.999:
        print("-> some of the study area has no DTM: download the missing tile(s)")
