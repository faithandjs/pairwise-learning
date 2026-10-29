"""Central settings. Edit the two input paths below; steps 2-5 read everything from here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---- input data (you put these files here) ----
OSM_PBF = ROOT / "data" / "raw" / "osm" / \
    "greater-manchester-261026.osm.pbf"  # Geofabrik extract, 1 Aug 2026
# folder with your LiDAR DTM tiles
DTM_DIR = ROOT / "data" / "raw" / "dtm"
# DTM_PATTERNS = ["*.tif", "*.tiff", "*.asc"]
DTM_PATTERNS = ["**/*.tif", "**/*.tiff", "**/*.asc"]
# only used if a tile stores no CRS (some .asc files)
DTM_CRS_IF_MISSING = "EPSG:27700"
GRADIENT_CEILING = 0.20

# ---- study area from the proposal (WGS84 lon/lat) ----
WEST, SOUTH, EAST, NORTH = -2.2974, 53.4609, -2.2748, 53.4744
PROJECTED_CRS = "EPSG:27700"        # British National Grid, metres

# ---- gradient settings ----
STEEP = 1 / 20        # 5 %     "steep" label
VERY_STEEP = 1 / 12   # 8.33 %  "very steep" label
WINDOW_M = 20.0       # smoothing window (metres) for gradient method B
MIN_RELIABLE_M = 10.0  # a gradient measured over less than this is flagged as unreliable

# ---- folders written by the scripts ----
INTERIM = ROOT / "data" / "interim"
OUTPUTS = ROOT / "outputs"
