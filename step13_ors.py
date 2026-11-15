"""Step 13: the openrouteservice (ORS) wheelchair baseline, scored the same way as every other method.

Run:  python step13_ors.py        (after step 12)
Reads:  outputs/od_pairs.csv, data/interim/segments.gpkg, outputs/routing_results.pkl (from step 12)
Needs:  an ORS API key (free at openrouteservice.org) in the environment variable ORS_API_KEY,
        or in a file called .env in this folder containing one line:  ORS_API_KEY=your_key_here
Writes: data/interim/ors/od_<id>.json   (the raw answers, kept so each journey is requested only once)
        outputs/ors_routes.csv          (one row per journey: found, distance, how well it matched our segments)
        outputs/routing_results.csv     (step 12's file with the ORS rows added)

How it works
  1. For each origin-destination pair, ask ORS for a route with the standard 'wheelchair' profile (default settings,
     as the supervisor asked). ORS uses its own, current OpenStreetMap data, not your frozen 26 Oct 2026 extract.
  2. Match the ORS line onto YOUR segments: a segment counts as part of the route if at least MIN_SHARE of its length lies
     within TOL metres of the ORS line AND both its ends are within TOL metres of the line (so short side streets
     that only touch the route are not counted). The matched segments are then measured exactly like the other methods' routes.
  3. Quality check: matched length / the distance ORS reports. If it is outside COVERAGE_OK the route is not scored
     (it may have left the study area or followed streets your network does not contain). These are counted and shown.
The ORS route does not depend on the profile (it is the standard profile), so the same route is scored against each
profile's bans and true weights.
"""
import json
import os
import time
import urllib.error
import urllib.request

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString, Point

import config
from common import split_journeys
from profiles import PROFILES
from routing_common import Segs, add_regret, print_summary, profile_weights, score_route, SPLIT, load_learner_inputs

TOL = getattr(config, "ORS_MATCH_TOL_M", 3.0)                  # Note-> metres either side of the ORS line
MIN_SHARE = getattr(config, "ORS_MATCH_MIN_SHARE", 0.8)        # Note-> share of a segment that must lie inside that band
COVERAGE_OK = getattr(config, "ORS_COVERAGE_OK", (0.85, 1.15))
PAUSE_S = 2.0                                                  # Note-> pause between requests (ORS limits how fast you can call)
URL = "https://api.openrouteservice.org/v2/directions/wheelchair/geojson"
CACHE = config.INTERIM / "ors"


def get_key():
    key = os.environ.get("ORS_API_KEY")
    env = config.ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("ORS_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    return key


def metres_apart_squared(lon1, lat1, lon2, lat2):
    """Rough squared distance in metres between two points; good enough at this scale."""
    return ((lon1 - lon2) * 111320 * np.cos(np.radians(lat1))) ** 2 + ((lat1 - lat2) * 110540) ** 2


def cache_matches(data, o, d):
    """Is this saved answer really for this journey? (Journey numbers change if step 8 is run again with other settings.)"""
    if "_request" in data:
        return data["_request"] == [round(o[0], 5), round(o[1], 5), round(d[0], 5), round(d[1], 5)]
    if "features" in data:                              # older saved answers: check the ends of the route are near the start and end
        c = data["features"][0]["geometry"]["coordinates"]
        near = lambda p, q: metres_apart_squared(p[0], p[1], q[1], q[0]) <= 400 ** 2     # ORS snaps to roads within 350 m
        return near(c[0], o) and near(c[-1], d)
    return True


def fetch(od_id, o, d, key):
    f = CACHE / f"od_{od_id}.json"
    if f.exists():
        saved = json.loads(f.read_text())
        if cache_matches(saved, o, d):
            return saved
        if not key:
            raise SystemExit(f"Saved ORS route for journey {od_id} belongs to a different journey (the pairs changed) "
                             "and no API key was found to fetch it again. Delete data/interim/ors or set ORS_API_KEY.")
    body = json.dumps({"coordinates": [[o[1], o[0]], [d[1], d[0]]]}).encode()      # ORS wants [longitude, latitude]
    req = urllib.request.Request(URL, data=body, method="POST",
                                 headers={"Authorization": key, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        data = {"error": f"HTTP {e.code}", "detail": e.read().decode(errors="replace")[:500]}
        if e.code in (401, 403):
            raise SystemExit(f"ORS refused the request (HTTP {e.code}): check the API key. Nothing was saved for this journey.")
        if e.code == 429:
            raise SystemExit("ORS says you are sending too many requests (HTTP 429). Routes already fetched are kept; "
                             "run the script again later and it will carry on.")
        if e.code not in (400, 404):                                                # Note-> server trouble: do not save, try again next run
            return data
    except Exception as e:                                                          # network problem: do not save it
        return {"error": f"request failed: {e}"}
    data["_request"] = [round(o[0], 5), round(o[1], 5), round(d[0], 5), round(d[1], 5)]
    f.write_text(json.dumps(data))
    time.sleep(PAUSE_S)
    return data


def parse(data):
    """Return (list of [lon, lat], distance in metres) from an ORS GeoJSON answer, or (None, reason)."""
    if "features" not in data:
        return None, data.get("error", "no route in answer")
    feat = data["features"][0]
    return feat["geometry"]["coordinates"], float(feat["properties"]["summary"]["distance"])


def match(geo, sindex, coords):
    tr = Transformer.from_crs(4326, config.PROJECTED_CRS, always_xy=True)
    line = LineString([tr.transform(x, y) for x, y, *_ in coords])
    band = line.buffer(TOL)
    cand = sindex.query(band, predicate="intersects")
    geoms = geo.geometry.iloc[cand]
    share = geoms.intersection(band).length.to_numpy() / geoms.length.to_numpy()
    # Note-> a short side street that merely touches the route is NOT part of it: both ends of the segment must be near the line
    ends = np.array([line.distance(Point(g.coords[0])) <= TOL and line.distance(Point(g.coords[-1])) <= TOL for g in geoms])
    return cand[(share >= MIN_SHARE) & ends]


def main():
    od = pd.read_csv(config.OUTPUTS / "od_pairs.csv")
    CACHE.mkdir(parents=True, exist_ok=True)
    key = get_key()
    def saved_ok(r):
        f = CACHE / f"od_{r.od_id}.json"
        return f.exists() and cache_matches(json.loads(f.read_text()), (r.origin_lat, r.origin_lon), (r.dest_lat, r.dest_lon))
    missing = [r.od_id for r in od.itertuples() if not saved_ok(r)]
    if missing and not key:
        raise SystemExit(f"{len(missing)} journeys still need an ORS route and no API key was found.\n"
                         "Put it in the environment variable ORS_API_KEY or in a file .env containing ORS_API_KEY=...")
    geo = gpd.read_file(config.INTERIM / "segments.gpkg")
    sindex = geo.sindex
    S = Segs()

    rows = []
    for r in od.itertuples():
        data = fetch(r.od_id, (r.origin_lat, r.origin_lon), (r.dest_lat, r.dest_lon), key)
        coords, dist = parse(data)
        row = dict(od_id=r.od_id, ors_found=coords is not None, ors_distance_m=dist if coords else np.nan,
                   reason="" if coords else dist, matched_m=np.nan, coverage=np.nan, scored=False, rows="")
        if coords:
            idx = match(geo, sindex, coords)
            row["matched_m"] = float(S.length[idx].sum())
            row["coverage"] = row["matched_m"] / dist if dist else np.nan
            row["scored"] = bool(len(idx) and COVERAGE_OK[0] <= row["coverage"] <= COVERAGE_OK[1])
            row["rows"] = ";".join(map(str, idx))
        rows.append(row)
    ors = pd.DataFrame(rows)
    ors.to_csv(config.OUTPUTS / "ors_routes.csv", index=False)
    print(f"ORS routes found for {int(ors.ors_found.sum())} of {len(ors)} journeys; "
          f"matched well enough to score: {int(ors.scored.sum())}")
    if (~ors.ors_found).any():
        print("Reasons routes were not found:", ors.loc[~ors.ors_found, "reason"].value_counts().to_dict())
    print(f"coverage (matched length / ORS distance): median {ors.coverage.median():.2f}, "
          f"range {ors.coverage.min():.2f} to {ors.coverage.max():.2f}")

    # ---- score the ORS route for every repeat, profile and test journey ----
    res = pd.read_pickle(config.OUTPUTS / "routing_results.pkl")
    res = res[res.method != "ors"]

    refs, _ = load_learner_inputs()
    all_ods = np.array(sorted(pd.read_csv(config.OUTPUTS / "candidates.csv")["od_id"].unique()))
    info = ors.set_index("od_id")
    new = []
    for seed, ref in refs.items():
        test = split_journeys(all_ods, seed, SPLIT)[2]
        for pid in PROFILES:
            w_true = profile_weights(pid)
            for o in test:
                i = info.loc[o]
                row = dict(seed=seed, profile=pid, noise="", method="ors", od_id=int(o), found=bool(i.ors_found),
                           scored=bool(i.scored), seconds=np.nan)
                if i.scored:
                    row.update(score_route(S, [int(x) for x in str(i.rows).split(";")], pid, w_true, ref))
                new.append(row)
    res = add_regret(pd.concat([res, pd.DataFrame(new)], ignore_index=True))
    res.to_csv(config.OUTPUTS / "routing_results.csv", index=False)
    res.to_pickle(config.OUTPUTS / "routing_results.pkl")
    print_summary(res)
    n_test = res[res.method == "ors"]
    print(f"\nORS rows: {len(n_test)}, found: {int(n_test.found.sum())}, scored (good match): {int(n_test.scored.sum())}. "
          "Only scored rows are averaged; ms_per_route is blank for ORS (it runs on ORS's servers, not here).")


if __name__ == "__main__":
    main()
