# Claude Code Spec — 3D Change Detection Streamlit App

## Context

We have an existing Python script (`change_detection.py` or similar) that runs a 3D point cloud change detection pipeline on two LAS files (Toulouse urban data, epoch 0 and epoch 1). The script currently uses Open3D for processing and visualization, outputs to Open3D pop-up windows, and saves PLY files.

We need to wrap this into a Streamlit web app for a demo video. Speed is the priority. Do not over-engineer.

---

## Task Overview

1. Refactor the existing script into a clean module (`pipeline.py`) — functions only, no script-level execution
2. Create a new Streamlit app (`app.py`) that calls those functions and displays results
3. Replace all Open3D visualizations with Plotly 3D scatter plots (browser-compatible)
4. Layout follows a specific 4-section structure described below

---

## File Structure

```
project/
├── app.py              # Streamlit app (NEW)
├── pipeline.py          # Refactored pipeline functions (NEW, from existing code)
├── requirements.txt     # Dependencies (NEW)
├── data/
│   ├── toulouse_lt0.las
│   └── toulouse_lt1.las
└── output/              # Generated outputs (existing)
```

---

## Step 1: Refactor existing code into `pipeline.py`

Take ALL the existing functions and put them in `pipeline.py` as a clean importable module. Keep every function exactly as-is:

- `load_point_cloud(file_path)` — returns Open3D point cloud
- `preprocess_point_cloud(pcd, voxel_size, remove_outliers)` — returns downsampled pcd
- `register_point_clouds(source, target, voxel_size, max_iter)` — returns aligned pcd + transformation
- `compute_cloud_distances(source, target)` — returns distances array
- `analyze_changes(distances, threshold)` — returns change_indices and stats dict
- `create_distance_heatmap(source, distances)` — returns colored pcd
- `detect_missing_regions(source, target, distances, distance_threshold, region_size_threshold)` — returns regions, missing_indices, region_labels
- `transfer_colors_to_original(original_pcd, colored_downsampled_pcd)` — returns colored pcd
- `save_results(...)` — saves PLY files and stats

DO NOT modify any pipeline logic. Just move functions into the module, remove all script-level code (the lines between `#%%` blocks that call functions directly).

Add one new helper function:

```python
def pcd_to_numpy(pcd):
    """Convert Open3D point cloud to numpy arrays for plotting"""
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    return points, colors
```

---

## Step 2: Create `app.py` — Streamlit App

### Page Config

```python
st.set_page_config(
    page_title="3D Change Detection — Toulouse",
    layout="wide"
)
```

### App Title

```
3D Urban Change Detection — Toulouse, France
```

Subtitle (smaller text):

```
Automated detection of structural changes between two LiDAR scans using AI-powered point cloud analysis
```

---

### Processing Flow

On app load (use `@st.cache_data` or `@st.cache_resource` to avoid recomputing on every interaction):

1. Load both LAS files
2. Preprocess both (voxel_size=0.1)
3. Register if needed
4. Compute distances
5. Analyze changes (threshold=0.15)
6. Create heatmap
7. Detect missing regions
8. Generate colored point cloud with regions

Show a `st.progress()` bar while processing, with status text updating at each step.

---

### Visualization: Plotly 3D Scatter

Create a reusable function for rendering point clouds in Plotly:

```python
def plot_point_cloud(points, colors=None, name="Point Cloud", point_size=1.5, max_points=50000):
    """
    Render a point cloud as a Plotly 3D scatter plot.
    
    - points: numpy array (N, 3)
    - colors: numpy array (N, 3) with values 0-1, or None for uniform color
    - max_points: downsample randomly to this many points for browser performance
    """
```

IMPORTANT: Always randomly subsample to max 50,000 points for Plotly rendering. Full point clouds will freeze the browser. Keep the same random seed so views are consistent.

For the color conversion to Plotly format:
```python
# Convert (N,3) float colors [0,1] to Plotly RGB strings
color_strings = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r, g, b in colors]
```

Set consistent camera angle across all plots so before/after are visually comparable. Use:
```python
camera = dict(
    eye=dict(x=1.5, y=1.5, z=1.0),
    up=dict(x=0, y=0, z=1)
)
fig.update_layout(scene_camera=camera)
```

Remove grid lines and axis labels for a cleaner look. Set dark background:
```python
fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        bgcolor='#0e1117'
    ),
    paper_bgcolor='#0e1117',
    margin=dict(l=0, r=0, t=0, b=0),
    height=500
)
```

---

### Layout: 4 Sections

#### SECTION 1 — THE RESULT (full width, top of page)

Heading: `## Detected Changes`

Show the final colored point cloud (grey = unchanged, colored by region = changed areas). Full width Plotly 3D scatter. This is the hero visual — the first thing anyone sees.

Below the plot, one line of text:
```
{len(regions)} change regions detected across {stats['volume_change_percentage']:.1f}% of the scanned area
```

---

#### SECTION 2 — BEFORE & AFTER (two columns)

Heading: `## Before & After`

Use `col1, col2 = st.columns(2)`

- Left column: header "Epoch 0 (Before)", Plotly 3D scatter of preprocessed epoch 0 in a uniform color (e.g., `rgb(100, 149, 237)` — cornflower blue)
- Right column: header "Epoch 1 (After)", Plotly 3D scatter of preprocessed epoch 1 in a different uniform color (e.g., `rgb(255, 127, 80)` — coral)

Same camera angle on both plots.

---

#### SECTION 3 — CHANGE REPORT (full width)

Heading: `## Change Report`

Use `st.columns(4)` to display key metrics as `st.metric()` cards:

- Column 1: "Points Analyzed" → total number of points in source cloud
- Column 2: "Change Regions" → number of detected regions
- Column 3: "Area Affected" → volume_change_percentage with "%" suffix
- Column 4: "Max Displacement" → max_change with "m" suffix

Below the metrics, add a horizontal bar chart (Plotly) showing the size of each detected region (number of points per region), sorted largest to smallest. Title: "Region Size Distribution". Keep it simple, horizontal bars, dark theme matching the rest of the app.

---

#### SECTION 4 — TRANSITION VIEWER (full width, bottom)

Heading: `## Transition Viewer`

A `st.radio()` with three options displayed horizontally: `["Before", "After", "Changes Highlighted"]`

- "Before" → shows epoch 0 point cloud (uniform blue)
- "After" → shows epoch 1 point cloud (uniform coral)
- "Changes Highlighted" → shows the heatmap point cloud (distance-based coloring)

All three use the SAME camera angle and plot dimensions. Only one is displayed at a time based on selection. This lets the viewer toggle between views in the demo video.

---

## Step 3: Heatmap Color Improvement

In `pipeline.py`, update the `create_distance_heatmap()` function to use matplotlib's colormap instead of manual RGB mixing:

```python
from matplotlib import cm

# Replace the manual color mixing with:
cmap = cm.get_cmap('RdYlBu_r')  # Red = high change, Blue = low change
colors = cmap(normalized_dists)[:, :3]  # Drop alpha channel
```

This is the ONE change allowed to pipeline logic — it's purely cosmetic.

---

## Step 4: requirements.txt

```
streamlit
open3d
laspy
numpy
scipy
matplotlib
plotly
```

---

## Constraints

- DO NOT add file upload functionality. Hardcode paths to `./data/toulouse_lt0.las` and `./data/toulouse_lt1.las`
- DO NOT add authentication or user management
- DO NOT create separate CSS/JS files. Keep everything in app.py
- DO NOT change any pipeline algorithm logic (voxel downsampling, ICP, KDTree, region growing — all stays exactly as-is)
- DO subsample points for Plotly rendering (max 50k per plot). Processing uses full resolution
- DO use @st.cache_resource for pipeline results so the app doesn't recompute on every widget interaction
- DO use Streamlit's dark theme by default (set in .streamlit/config.toml if needed)
- Keep the code clean and minimal. No unnecessary comments or verbose variable names

---

## Config File

Create `.streamlit/config.toml`:

```toml
[theme]
base = "dark"
primaryColor = "#4FC3F7"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#1a1a2e"
textColor = "#ffffff"
```

---

## Testing

After building, verify:
1. `streamlit run app.py` launches without errors
2. Processing progress bar works
3. All 4 sections render with Plotly 3D plots
4. Transition viewer toggles between 3 views
5. Plots don't freeze the browser (check point count is capped at 50k per plot)
6. Camera angles are consistent across all plots