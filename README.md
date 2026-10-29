# An AI-Based Personalised Route Optimisation Framework for Accessible Urban Navigation

COM748 Masters Research Project, Ulster University.

This project asks: to what extent can pairwise preference learning recover simulated, user-specific
accessibility trade-offs, and produce wheelchair-accessible pedestrian routes that beat generic routing
profiles on unseen route-choice scenarios? The preferences are simulated, not taken from real wheelchair users.

Study area: a 1.5 km cell covering Salford Quays, the Trafford Road corridor, Old Trafford and Pomona, Manchester
(bounding box 53.4609-53.4744 N, 2.2974-2.2748 W).

## Status

Done: the data foundation (Phase 0, data audit, plus building the per-segment features). This folder contains
only that part.

Not built yet: simulated profiles, candidate-route generation, synthetic choices, preference learning, routing,
baselines, evaluation and the explanation layer.

Known open issue: gradients from the LiDAR terrain model are wrong around bridges and similar structures.
Ways tagged `bridge` or `tunnel` in OpenStreetMap now have their gradient ignored, but a check on the 24 steepest
streets found only 7 tagged that way (all with `layer=1`); 17 were untagged and still show gradients of 38 to 82%.
A further fix (for example a ceiling for impossible gradients, or a list of excluded ways) is still to be decided.

## Data

| Data                                    | Source                                                                                                                  | Licence / attribution                |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| Street network, steps, crossings, kerbs | OpenStreetMap, Greater Manchester extract from Geofabrik (file header timestamp 2026-10-26T21:24:46Z)                   | ODbL. (c) OpenStreetMap contributors |
| Terrain height                          | Environment Agency / DEFRA LiDAR Composite DTM, 1 m resolution, 5 km x 5 km tiles in British National Grid (EPSG:27700) | Open Government Licence              |

The raw data is not stored in this repository (it is large and can be re-downloaded). Put it here:

    data/raw/osm/greater-manchester-261026.osm.pbf     (or any name; set OSM_PBF in config.py)
    data/raw/dtm/                                      (tiles may sit in subfolders)

## Folder structure

    config.py                 settings: file paths, study-area box, gradient thresholds
    requirements.txt          library versions
    step1_check_inputs.py     checks libraries, OSM file, DTM tiles and study-area coverage
    step2_load_network.py     builds the walkable street network from the OSM file
    step3_audit_osm.py        counts steps, kerbs, crossings and tags (the data audit)
    step4_gradient.py         attaches a gradient to each street segment from the DTM
    step5_segment_table.py    builds the one-row-per-segment table
    tests/make_test_data.py   optional: makes fake data to test the code (see warning below)
    data/raw/                 original downloads, never edited
    data/interim/             files passed between steps; safe to delete and regenerate
    outputs/                  results to read (maps, audit tables, the segment table)

## Setup (once)

macOS / Linux:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt

Windows (PowerShell):

    py -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt

Developed with Python 3.13; library versions are pinned in requirements.txt. If pip cannot install geopandas,
rasterio or pyrosm, use conda-forge instead:

    conda create -n arp python=3.12
    conda activate arp
    conda install -c conda-forge geopandas rasterio pyrosm scipy matplotlib
    pip install osmium

If a script says a library is not found, you are probably not in the environment. Run `which python`; the path
should end in `.venv/bin/python`.

## Run (in order, from this folder, with the environment active)

    python step1_check_inputs.py
    python step2_load_network.py
    python step3_audit_osm.py
    python step4_gradient.py
    python step5_segment_table.py

| Step | What it does                                                                                                               | Main outputs                                                                     |
| ---- | -------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| 1    | Checks installed libraries, the OSM file, DTM coordinate system and study-area coverage                                    | printed report                                                                   |
| 2    | Loads the walkable network for the study area, projected to EPSG:27700                                                     | data/interim/edges.gpkg, nodes.gpkg; outputs/network_map.png                     |
| 3    | Audits steps, step counts, incline, surface, wheelchair, smoothness tags; crossings; kerb and tactile-paving tags on nodes | outputs/audit_summary.csv; data/interim/node_tags.csv                            |
| 4    | Samples the DTM at each node and computes a gradient per segment                                                           | data/interim/edges_gradient.gpkg; outputs/gradient_report.csv, gradient_hist.png |
| 5    | Builds the final table of the four route features per segment                                                              | outputs/segment_table.csv; data/interim/segments.gpkg                            |

Re-run step 2 after changing the OSM file or study area, and steps 4 and 5 after changing gradient settings.

## The four route features (per street segment)

Each feature adds up along a route, so it can later become a street-by-street routing cost.

| Feature  | Column           | Definition                                                                                                                                                                                                 |
| -------- | ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Distance | `length_m`       | Segment length in metres                                                                                                                                                                                   |
| Gradient | `steep_len_m`    | Metres of segment steeper than 1 in 20 (5%), using the 20 m windowed gradient. Absolute gradient (up and down count the same). Zero on steps, and zero where the gradient is ignored (bridges and tunnels) |
| Steps    | `steps_length_m` | Metres of segment tagged `highway=steps`; `step_count_tag` keeps the OSM step count where present                                                                                                          |
| Kerb     | `kerb_units`     | 1 per crossing not confirmed as lowered or flush (judged from the kerb tags on the crossing's two end nodes), shared equally across the crossing's segments                                                |

Other useful columns: `grade_a` (end-to-end gradient of the segment), `grade_b` (gradient over a 20 m stretch of
the same OSM way, centred on the segment; the main method), `grade_reliable` (False if measured over under 10 m),
`gradient_status` (`measured`, `excluded_bridge_or_tunnel` or `no_elevation`), `on_structure`, `kerb_status`
(`not_crossing`, `ok`, `raised`, `unknown`, `other`) and `kerb_raised_known` (a known barrier for hard constraints).

Reference gradient labels (labels, not universal limits): preferred is 1 in 20 or gentler; steep is steeper than
1 in 20; very steep is steeper than 1 in 12 (about 8.33%). The 1 in 12 figure comes from ramp-design guidance
(DfT Inclusive Mobility) and is not a boundary for every pedestrian route.

Missing information is kept as unknown, never assumed accessible. Settings for the gradient window and the
reliability flag are in `config.py`; the kerb rule is in `step5_segment_table.py`.

## What the audit found so far

- 8,890 segments, 101.6 km of walkable network; 68% of segments are shorter than 10 m.
- Steps: 106 segments in 73 flights (489 m); 35 of the 73 flights have a step count.
- Tag coverage as share of network length: incline 0.4%, surface 79.2%, wheelchair 0.8%, smoothness 1.3%.
- Kerb tags on 392 nodes: 290 lowered, 87 raised, 15 flush. 761 nodes have a tactile-paving tag.
- Of 238 crossings: 109 confirmed fine, 40 with a raised kerb, 89 unknown (about 37%).

Gradient results are in `outputs/gradient_report.csv` (regenerated each run).

## Checking the steepest segments

Open `outputs/gradient_hist.png` and `outputs/gradient_report.csv`. The line "segments STILL steeper than 20%
after that (check these)" in the step 4 printout counts segments that are still implausibly steep after bridges and
tunnels are ignored. Look up suspicious ways on openstreetmap.org (Query features) to see their tags.

## Optional: test the code on fake data

`tests/make_test_data.py` creates a small fake OSM file and fake DTM tiles with known answers (listed at the top of
the script). It refuses to run if real data is already in place, so it cannot overwrite your files, but only
use it in an empty copy of the project. Delete the fake files, data/interim and outputs before adding real data.

## Planned next

Five simulated accessibility profiles (hard constraints and weights); candidate-route generation by penalising
used streets; synthetic pairwise choices with noise; a pairwise preference learner (RankSVM or logistic ranking);
routing with learned edge costs; baselines (shortest path, openrouteservice wheelchair profile, a generic
hand-weighted profile, and an oracle using the true weights); evaluation on held-out origin-destination pairs;
and a plain-language explanation layer.

This is a research prototype and planning aid, not a safety-certified navigation system.
