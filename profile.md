# Simulated accessibility profiles (draft)

Status: DRAFT. Eight simulated users. Weights and bans are my design choices to give the preference-learning model a
known answer to recover. They are not measurements of real wheelchair users and have no empirical basis; the
report must say so. To be confirmed after the feasibility check (Section 7).

## 1. How to read a profile

Each profile has two parts.

Hard bans: segments the profile cannot use. They are removed from that profile's network before any route is found.

Weights: how much the profile dislikes each of the four route features, as shares that sum to 1.
- A higher weight means a bigger penalty. The feature with the highest weight is the one the profile cares most about
  avoiding. Nothing is rewarded.
- Route cost = sum over the four features of (weight x normalised feature value). The simulated user prefers the
  route with the lower cost (equivalently, higher utility = minus cost), with some randomness added.
- Each feature is first rescaled to 0 to 1 using fixed study-area reference values (methodology point 6). These
  reference values are not chosen yet, so until they are, weights show relative importance within a profile and are
  not exchange rates (0.45 on steps versus 0.10 on distance does not mean one step equals 4.5 metres).
- A feature that a profile bans has no weight for that profile, because every allowed route has zero of it.

## 2. The four features (columns in outputs/segment_table.csv)

| Feature | Column | Meaning |
| --- | --- | --- |
| Distance | length_m | Metres walked |
| Gradient | steep_len_m | Metres on segments steeper than 1 in 20 (5%), 20 m windowed gradient (grade_b); zero on steps and where the gradient is unknown |
| Steps | steps_length_m | Metres of steps (highway=steps) |
| Kerb | kerb_units | 1 per crossing not confirmed as lowered or flush; unknown counts as a cost, never as accessible |

## 3. The ban rules

| Ban | Segment is removed if | Notes |
| --- | --- | --- |
| Steps | steps_is_steps is true | 106 segments, 73 flights, 489 m in the study area |
| Known raised kerbs | kerb_raised_known is true | 40 of 238 crossings. Unknown kerbs are NOT banned |
| Steeper than 1 in 12 | grade_b above 0.0833 | About 0.8% of network length (20% ceiling run) |
| Steeper than 1 in 20 | grade_b above 0.05 | About 2.5% of network length (20% ceiling run) |

The 1 in 12 and 1 in 20 figures come from ramp-design guidance (DfT Inclusive Mobility) as reported by secondary
sources [VERIFY against the primary document]. They are used as reference limits for simulated users, not as universal
accessibility boundaries.

## 4. The eight profiles

| # | Profile | Hard bans | Weights (sum to 1) |
| --- | --- | --- | --- |
| 1 | Self-propelling, effort-sensitive | steps, known raised kerbs, steeper than 1 in 12 | gradient .50, distance .25, kerb .25 |
| 2 | Powered chair, kerb-sensitive | steps, known raised kerbs | kerb .55, distance .25, gradient .20 |
| 3 | Low stamina, distance-sensitive | steps, steeper than 1 in 12 | distance .60, gradient .30, kerb .10 |
| 4 | Everyday manual user, balanced | steps | kerb .40, distance .35, gradient .25 |
| 5 | Strict, avoids slopes | steps, known raised kerbs, steeper than 1 in 20 | distance .40, kerb .40, gradient .20 |
| 6 | Companion-assisted | none | steps .40, kerb .25, distance .20, gradient .15 |
| 7 | Barrier-tolerant | none | distance .40, gradient .25, steps .20, kerb .15 |
| 8 | Barrier-averse, no walls | none | steps .45, kerb .35, distance .10, gradient .10 |

## 5. What each profile represents

1. Pushes their own chair, tires on slopes: gradient is the main worry; bans steps, raised kerbs and very steep ground.
2. Powered chair, copes with slopes and distance but depends on dropped kerbs: kerb weight dominates.
3. Limited stamina: distance matters most, then slopes.
4. A middle case that cares about kerbs and distance but is not extreme on any feature.
5. Stricter slope limit (1 in 20) than profiles 1 and 3 (1 in 12), with kerbs and distance as the main costs; tests the
   user-specified maximum gradient in the methodology.
6. Has a companion who can help over a short flight of steps or a kerb: steps are a heavy cost, not a wall.
7. Treats barriers as effort rather than walls and mainly wants a short route.
8. Strongly dislikes steps and kerbs but has no ban, which shows what soft avoidance does compared with strict
   avoidance (the literature notes strict avoidance can make journeys impossible [6], [7]).

## 6. Design principles and why the weights are not equal

- The profiles span different trade-offs so the learner is tested on more than one pattern.
- Distance, gradient and kerb are weighted costs in all or most profiles. Steps are a weighted cost only in
  profiles 6 to 8, because steps are banned elsewhere and a banned feature cannot have a learned weight.
- No profile uses equal weights: equal weights are the "generic hand-weighted" baseline, and a profile identical to
  the baseline could not be beaten by learning.
- Profiles 6 to 8 weight steps heavily instead of banning them. This is a modelling device that allows the steps
  weight to be learned, not a claim that wheelchair users climb stairs.

## 7. Limitations and checks still to do

- Weights are invented, as above. Results show whether the method recovers known weights, not what real users want.
- Steps are rare (73 flights), so few routes contain steps; the steps weight may be hard to recover even for
  profiles 6 to 8. Check how often candidate routes contain steps.
- Gradient is mostly flat in the area (90% of segments under 3.4%), so routes may differ little in gradient, which
  could make the gradient weight hard to recover. Check how much candidate routes differ in steep length.
- Gradient bans apply only where a gradient was measured. Segments set to unknown (bridges and tunnels, above the 20%
  ceiling, no elevation: 3.2% of segments) are not banned, so a genuinely steep one among them would not be excluded.
- Constraint violations can only occur for profiles 1 to 5; profiles 6 to 8 have no bans.
- Feasibility check done (step 6, 300 pairs, provisional until rerun): profiles 1 to 5 lose connectivity under
  gradient and kerb bans; profile 5 keeps 72.1% of the network connected and has no route for 57.7% of pairs.
  See decision-logs/02_profile_layer.md.
- Normalisation reference values and choice-noise levels are now set (decision-logs/03_route_choice_layer.md).
- Banned features do not vary within a profile's routes: steps weight is unrecoverable for profiles 1 to 5, gradient
  weight for profile 5.
