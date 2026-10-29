# Accessible routing: data foundation (steps 1-5)

## Set up (once)
macOS / Linux:
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Windows (PowerShell):
    py -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -r requirements.txt

If pip cannot install geopandas / rasterio / pyrosm on your machine, use conda-forge instead
(conda create -n arp python=3.12; conda activate arp; conda install -c conda-forge geopandas rasterio pyrosm scipy matplotlib; pip install osmium).

## Put your data in place
    data/raw/osm/greater-manchester.osm.pbf     <- the Geofabrik extract (or edit OSM_PBF in config.py)
    data/raw/dtm/                               <- your LiDAR DTM tiles (.tif / .tiff / .asc)

## Run, in order (always from this folder, with the environment active)
    python step1_check_inputs.py     # libraries, files, tile CRS and study-area coverage
    python step2_load_network.py     # street network -> data/interim, outputs/network_map.png
    python step3_audit_osm.py        # OSM audit -> outputs/audit_summary.csv
    python step4_gradient.py         # gradient from DTM -> outputs/gradient_report.csv + histogram
    python step5_segment_table.py    # outputs/segment_table.csv (one row per street segment)

## Optional: test the code on synthetic data first
    python tests/make_test_data.py   # writes a small fake OSM file + DTM tiles into data/raw
    python step1_check_inputs.py  ... through step 5
Then delete data/raw/osm/greater-manchester.osm.pbf, data/raw/dtm/test_tile_*.tif, data/interim/* and outputs/*
before adding your real data. Known truth in the fake data is listed at the top of tests/make_test_data.py.
