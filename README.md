# 3D change detection

Find what changed between two 3D scans of the same place.

Demo video: https://youtu.be/7ndOr8IyHq0

## What it does

You give it two point clouds (LAS files) of the same place, taken at different times. It shows you which parts changed.

1. **Load and clean.** Scale each scan down on a voxel grid and remove stray dots.
2. **Line up.** Move the second scan onto the first one (ICP, point to plane).
3. **Measure.** For every dot in the first scan, find the nearest dot in the second scan and measure the distance (KD-tree).
4. **Flag.** A dot that is far from the other scan has changed. The limit is yours to choose. This repo uses 10 to 15 cm.
5. **Show.** Color the dots by distance (heat map) and group the changed dots into regions.

## The data

Two scans of a tower with a wall and trees, in Toulouse, France. The area is small, about 23 m wide.

| File | Scan |
|---|---|
| `data/toulouse_lt0.las` | before (scan 1) |
| `data/toulouse_lt1.las` | after (scan 2) |

The scans are public data from the French government. Check the licence of the original source before you reuse them.

## Quick start

```bash
pip install open3d laspy scipy numpy matplotlib
```

Tested with Python 3.13, Open3D 0.20 and laspy 2.7.

```python
import copy
import pipeline as P

# 1. Load and scale down both scans
before = P.preprocess_point_cloud(P.load_point_cloud("data/toulouse_lt0.las"), voxel_size=0.1)
after  = P.preprocess_point_cloud(P.load_point_cloud("data/toulouse_lt1.las"), voxel_size=0.1)

# 2. (Optional) line up the second scan. This changes the scan it is given, so pass a copy.
after_aligned, transform = P.register_point_clouds(copy.deepcopy(after), before, voxel_size=0.1)

# 3. Distance from every dot of scan 1 to the nearest dot of scan 2
distances = P.compute_cloud_distances(before, after_aligned)

# 4. Flag the dots that are more than 15 cm away
changed, stats = P.analyze_changes(distances, threshold=0.15)
print(stats)
```

## Files

| File | What it is |
|---|---|
| `index.py` | The full step-by-step script, with the plots. |
| `index original.py` | A backup copy of `index.py`. |
| `pipeline.py` | The same steps as plain functions, with no script code. |
| `claude code spec.md` | The spec used to turn the pipeline into a Streamlit demo. |

## What I found on this pair

These are my own checks. The two scans have no answer sheet.

- About 22% of the dots in scan 1 are more than 15 cm from scan 2.
- The pointed roof of the tower is gone in scan 2. It is about 1 in 10 of the changed dots.
- About 6 in 10 of the changed dots sit in one tree. Leaves move and thin out, so a tree looks like a big change. The numbers cannot tell a roof from leaves. You have to look.
- Below 10 cm the result is mostly scan noise: at a 5 cm limit, 52% of the dots look changed. From 10 cm up, the share stays between 21% and 24%.
- The two scans were already lined up within about 5 cm. Lining them up again changed the share by about half a point.

## Limits

- **One direction.** Distances go from scan 1 to scan 2, so the pipeline finds what disappeared. To find what is new, swap the two scans and run it again. On this pair almost nothing is new.
- **Trees look like change.** Leaves, cars and people move between scans. Read the result by eye before you trust it.
- **The limit is a choice.** 15 cm worked here. A different place or scanner needs a different limit.
- **Small area.** This is one tower, not a street or a city.
- `volume_change_percentage` in the stats is the share of dots that changed, not a real volume.

## Credit

- Data and processing flow: Florent Poux (3D Geodata Academy). The scans are public data from the French government.
- Streamlit demo interface and the before and after transition: Giovanni Bazie.
