"""The eight simulated profiles (see profile.md). Later steps import this file so the profiles are defined once.

bans    : segments the profile cannot use. Removed from that profile's network before routing.
weights : how much the profile dislikes each feature (higher = bigger penalty). Sum to 1.
"""
import pandas as pd

import config

PROFILES = {
    1: dict(name="Self-propelling, effort-sensitive", bans=["steps", "raised_kerb", "grade_12"],
            weights=dict(gradient=0.50, distance=0.25, kerb=0.25)),
    2: dict(name="Powered chair, kerb-sensitive", bans=["steps", "raised_kerb"],
            weights=dict(kerb=0.55, distance=0.25, gradient=0.20)),
    3: dict(name="Low stamina, distance-sensitive", bans=["steps", "grade_12"],
            weights=dict(distance=0.60, gradient=0.30, kerb=0.10)),
    4: dict(name="Everyday manual user, balanced", bans=["steps"],
            weights=dict(kerb=0.40, distance=0.35, gradient=0.25)),
    5: dict(name="Strict, avoids slopes", bans=["steps", "raised_kerb", "grade_20"],
            weights=dict(distance=0.40, kerb=0.40, gradient=0.20)),
    6: dict(name="Companion-assisted", bans=[],
            weights=dict(steps=0.40, kerb=0.25, distance=0.20, gradient=0.15)),
    7: dict(name="Barrier-tolerant", bans=[],
            weights=dict(distance=0.40, gradient=0.25, steps=0.20, kerb=0.15)),
    8: dict(name="Barrier-averse, no walls", bans=[],
            weights=dict(steps=0.45, kerb=0.35, distance=0.10, gradient=0.10)),
}

for _k, _p in PROFILES.items():
    assert abs(sum(_p["weights"].values()) - 1.0) < 1e-9, f"profile {_k} weights do not sum to 1"


def banned_mask(seg, bans):
    """True for every segment this profile cannot use. Unknown gradient or unknown kerb is never banned."""
    m = pd.Series(False, index=seg.index)
    if "steps" in bans:
        m |= seg["steps_is_steps"].astype(bool)
    if "raised_kerb" in bans:
        m |= seg["kerb_raised_known"].astype(bool)
    if "grade_12" in bans:
        m |= (seg["grade_b"] > config.VERY_STEEP).fillna(False).astype(bool)
    if "grade_20" in bans:
        m |= (seg["grade_b"] > config.STEEP).fillna(False).astype(bool)
    return m
