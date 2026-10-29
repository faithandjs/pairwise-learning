"""Optional: create a small SYNTHETIC OSM file and DTM tiles so you can check your setup before using real data.

Run from the project folder:  python tests/make_test_data.py
It writes data/raw/osm/greater-manchester.osm.pbf and data/raw/dtm/test_tile_*.tif.
Delete those two files (and data/interim, outputs) before putting your real data in.

Known truth in the synthetic data:
  - east-west streets rise at exactly 6 % (steeper than 1 in 20); north-south footways are ~flat
  - 1 flight of steps (14 steps) on one north-south footway
  - 4 crossings: lowered/lowered (ok), flush/lowered (ok), raised/lowered (raised), no kerb tags (unknown)
  - DTM noise of 0.1 m (standard deviation) and a no-data hole in one corner
"""
import sys
from pathlib import Path

import numpy as np
import osmium
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

N = 60                      # sub-segments per street block (about 5 m each)
NB = 5                      # blocks in each direction
lons = np.linspace(config.WEST, config.EAST, NB * N + 1)
lats = np.linspace(config.SOUTH, config.NORTH, NB * N + 1)

node_ids, nodes_out, ways_out = {}, [], []
_next = [1000]


def nid(a, b):
    if (a, b) not in node_ids:
        node_ids[(a, b)] = _next[0]
        nodes_out.append((_next[0], lons[a], lats[b], {}))
        _next[0] += 1
    return node_ids[(a, b)]


wid = [5000]


def way(nodes, tags):
    ways_out.append((wid[0], nodes, tags))
    wid[0] += 1
    return wid[0] - 1


# east-west streets
for j in range(NB + 1):
    way([nid(a, j * N) for a in range(NB * N + 1)], {"highway": "residential"})
# north-south footways (line i=2 split so one piece is steps)
for i in range(NB + 1):
    if i == 2:
        way([nid(i * N, b) for b in range(0, N + 1)], {"highway": "footway"})
        way([nid(i * N, b) for b in range(N, N + 11)], {"highway": "steps", "step_count": "14"})
        way([nid(i * N, b) for b in range(N + 10, NB * N + 1)], {"highway": "footway"})
    else:
        way([nid(i * N, b) for b in range(NB * N + 1)], {"highway": "footway"})

# crossings: three nodes each (end, middle on the road, end)
def crossing(a_mid, j, kerb_a, kerb_b, label_id):
    lon, lat = lons[a_mid], lats[j * N]
    ids = []
    for k, (dlat, kerb) in enumerate(((0.00006, kerb_a), (0.0, None), (-0.00006, kerb_b))):
        tags = {}
        if k == 1:
            tags["highway"] = "crossing"
        if kerb:
            tags["kerb"] = kerb
        node_id = 900000 + label_id * 10 + k
        nodes_out.append((node_id, lon, lat + dlat, tags))
        ids.append(node_id)
    way(ids, {"highway": "footway", "footway": "crossing"})


crossing(30, 1, "lowered", "lowered", 1)
crossing(90, 2, "flush", "lowered", 2)
crossing(150, 3, "raised", "lowered", 3)
crossing(210, 4, None, None, 4)

out = Path(config.OSM_PBF)
out.parent.mkdir(parents=True, exist_ok=True)
if out.exists():
    out.unlink()
w = osmium.SimpleWriter(str(out))
for i, lon, lat, tags in sorted(nodes_out):
    w.add_node(osmium.osm.mutable.Node(id=i, location=(float(lon), float(lat)), tags=tags, version=1))
for i, nodes, tags in sorted(ways_out):
    w.add_way(osmium.osm.mutable.Way(id=i, nodes=nodes, tags=tags, version=1))
w.close()
print("wrote", out)

# ---- synthetic DTM tiles in British National Grid ----
tr = Transformer.from_crs("EPSG:4326", config.PROJECTED_CRS, always_xy=True)
x0, y0 = tr.transform(config.WEST, config.SOUTH)
x1, y1 = tr.transform(config.EAST, config.NORTH)
minx, maxx = min(x0, x1) - 100, max(x0, x1) + 100
miny, maxy = min(y0, y1) - 100, max(y0, y1) + 100
minx, miny = np.floor(minx), np.floor(miny)
maxx, maxy = np.ceil(maxx), np.ceil(maxy)
xmid = np.floor((minx + maxx) / 2)
rng = np.random.default_rng(42)
dtm_dir = Path(config.DTM_DIR)
dtm_dir.mkdir(parents=True, exist_ok=True)
for name, left, right in (("a", minx, xmid), ("b", xmid, maxx)):
    w_px, h_px = int(right - left), int(maxy - miny)
    xs = left + 0.5 + np.arange(w_px)
    z = 20 + 0.06 * (xs[None, :] - minx) + rng.normal(0, 0.1, size=(h_px, w_px))
    z = z.astype("float32")
    if name == "b":
        z[:250, -250:] = -9999          # no-data hole in the north-east corner (rows count from the top)
    with rasterio.open(dtm_dir / f"test_tile_{name}.tif", "w", driver="GTiff", height=h_px, width=w_px, count=1,
                       dtype="float32", crs=config.PROJECTED_CRS, transform=from_origin(left, maxy, 1.0, 1.0),
                       nodata=-9999) as dst:
        dst.write(z, 1)
print("wrote DTM tiles in", dtm_dir)
