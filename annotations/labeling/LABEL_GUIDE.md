# Label guide — activities & apparent age

Label **apparent** behavior and age from video only. Do not infer identity or true age.

## Step 1 — Is this a real person? (`label_status`)

Many clips are YOLO mistakes (poles, shadows, distant blobs). **Check this first.**

| `label_status` | When to use | What to fill |
|----------------|-------------|----------------|
| **`valid`** | Clear person in the crops | `activity_label` + `apparent_age_group` |
| **`not_a_person`** | Not a person (false detection, object, empty crop) | Leave activity/age **empty** |
| **`skip`** | Might be a person but too blurry / tiny / occluded to judge | Leave activity/age **empty** |

Examples of **`not_a_person`**: lamp post, tree trunk, bench, glare, fragment of a person (<10% of box), wrong object.

Examples of **`skip`**: person probably there but you cannot see posture or age.

**Do not** guess `walking` / `adult` on non-person clips — that poisons training data.

## Step 2 — Activity labels (only if `label_status` = `valid`)

| Label | When to use |
|-------|-------------|
| `walking` | Upright, steady locomotion on path/grass |
| `running` | Clearly faster than walking |
| `biking` | On bicycle (rider + bike visible) |
| `sitting` | Seated posture, little movement |
| `standing` | Upright, little or no movement, alone |
| `talking_socializing` | 2+ people close, interacting, slow or still |
| `playing` | Recreational movement (games, playful running, etc.) |
| `dog_walking` | Walking with a visible dog nearby |
| `exercising` | Workout-like motion (stretching, reps, sport drills) |
| `picnic_resting` | Stationary on grass/pavilion, resting or picnic setup |

If two apply, choose the **dominant** behavior in the clip.

## Step 3 — Age (only if `label_status` = `valid`)

| Label | When to use |
|-------|-------------|
| `child` | Apparent child/teen (smaller stature, proportions) |
| `adult` | Apparent adult |

Use **adult** only when you see a person but age is unclear at distance — not for `not_a_person` clips.

## manifest.csv columns

| Column | Required |
|--------|----------|
| `label_status` | Yes — `valid` / `not_a_person` / `skip` |
| `activity_label` | Only when `valid` |
| `apparent_age_group` | Only when `valid` |
| `notes` | Optional — e.g. "shadow", "only legs visible" |

## Quality

- 10% double-labeling recommended for agreement checks.
- Sort by `mean_confidence` / `mean_bbox_height_px` in manifest — review low values first.
