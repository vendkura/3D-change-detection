import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.spatial import KDTree
import laspy
import os


def load_point_cloud(file_path):
    if file_path.endswith('.las') or file_path.endswith('.laz'):
        las = laspy.read(file_path)
        points = np.vstack((las.x, las.y, las.z)).transpose()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
    elif file_path.endswith('.ply'):
        pcd = o3d.io.read_point_cloud(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    print(f"Loaded {len(pcd.points)} points from {file_path}")
    return pcd


def preprocess_point_cloud(pcd, voxel_size=0.05, remove_outliers=True):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    if remove_outliers:
        pcd_down, _ = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"Preprocessed cloud has {len(pcd_down.points)} points")
    return pcd_down


def register_point_clouds(source, target, voxel_size=0.05, max_iter=100):
    init_transformation = np.identity(4)
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6,
        relative_rmse=1e-6,
        max_iteration=max_iter
    )
    result = o3d.pipelines.registration.registration_icp(
        source, target, voxel_size*2, init_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria
    )
    source_transformed = source.transform(result.transformation)
    print(f"Registration finished with fitness: {result.fitness}, RMSE: {result.inlier_rmse}")
    return source_transformed, result.transformation


def compute_cloud_distances(source, target):
    target_points = np.asarray(target.points)
    source_points = np.asarray(source.points)
    tree = KDTree(target_points)
    distances, _ = tree.query(source_points)
    print(f"Computed distances for {len(source_points)} points")
    return distances


def analyze_changes(distances, threshold=0.1):
    change_indices = np.where(distances > threshold)[0]
    change_distances = distances[change_indices]
    if len(change_distances) > 0:
        mean_change = np.mean(change_distances)
        max_change = np.max(change_distances)
        total_volume_change = len(change_indices) / len(distances)
        print(f"Detected {len(change_indices)} points with significant change")
        print(f"Mean change: {mean_change:.3f}m, Max change: {max_change:.3f}m")
        print(f"Approximate volume change: {total_volume_change*100:.2f}%")
        return change_indices, {
            "mean_change": mean_change,
            "max_change": max_change,
            "volume_change_percentage": total_volume_change * 100
        }
    else:
        print("No significant changes detected")
        return [], {"mean_change": 0, "max_change": 0, "volume_change_percentage": 0}


def create_distance_heatmap(source, distances):
    heatmap_pcd = o3d.geometry.PointCloud()
    heatmap_pcd.points = o3d.utility.Vector3dVector(np.asarray(source.points))
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    if max_dist > min_dist:
        normalized_dists = (distances - min_dist) / (max_dist - min_dist)
    else:
        normalized_dists = np.ones_like(distances) * 0.5
    cmap = cm.get_cmap('RdYlBu_r')
    colors = cmap(normalized_dists)[:, :3]
    heatmap_pcd.colors = o3d.utility.Vector3dVector(colors)
    print(f"Heatmap color scale: Blue = {min_dist:.3f}m, Red = {max_dist:.3f}m")
    return heatmap_pcd


def detect_missing_regions(source, target, distances, distance_threshold=0.1, region_size_threshold=10):
    source_points = np.asarray(source.points)
    missing_indices = np.where(distances > distance_threshold)[0]
    if len(missing_indices) == 0:
        print("No significant differences detected")
        return [], [], []
    source_tree = KDTree(source_points)
    all_regions = []
    processed = np.zeros(len(source_points), dtype=bool)
    for idx in missing_indices:
        if processed[idx]:
            continue
        current_region = [idx]
        processed[idx] = True
        i = 0
        while i < len(current_region):
            current_idx = current_region[i]
            neighbors_dist, neighbors_idx = source_tree.query(
                source_points[current_idx].reshape(1, -1), k=20)
            for neighbor_idx in neighbors_idx[0][1:]:
                if not processed[neighbor_idx] and neighbor_idx in missing_indices:
                    current_region.append(neighbor_idx)
                    processed[neighbor_idx] = True
            i += 1
        if len(current_region) >= region_size_threshold:
            all_regions.append(current_region)
    all_missing_indices = []
    region_labels = np.zeros(len(source_points), dtype=int)
    for region_idx, region in enumerate(all_regions, 1):
        all_missing_indices.extend(region)
        for point_idx in region:
            region_labels[point_idx] = region_idx
    print(f"Detected {len(all_regions)} missing regions with total {len(all_missing_indices)} points")
    return all_regions, np.array(all_missing_indices), region_labels


def transfer_colors_to_original(original_pcd, colored_downsampled_pcd):
    colored_original = o3d.geometry.PointCloud()
    colored_original.points = o3d.utility.Vector3dVector(np.asarray(original_pcd.points))
    original_points = np.asarray(original_pcd.points)
    downsampled_points = np.asarray(colored_downsampled_pcd.points)
    downsampled_colors = np.asarray(colored_downsampled_pcd.colors)
    tree = KDTree(downsampled_points)
    _, indices = tree.query(original_points)
    original_colors = downsampled_colors[indices]
    colored_original.colors = o3d.utility.Vector3dVector(original_colors)
    print(f"Transferred colors from downsampled cloud ({len(downsampled_points)} points) to original cloud ({len(original_points)} points)")
    return colored_original


def save_results(colored_pcd, missing_pcd, heatmap_pcd, regions, stats, output_dir="./output"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if colored_pcd is not None:
        o3d.io.write_point_cloud(f"{output_dir}/colored_source.ply", colored_pcd)
        print(f"Saved colored source point cloud to {output_dir}/colored_source.ply")
    if missing_pcd is not None and len(missing_pcd.points) > 0:
        o3d.io.write_point_cloud(f"{output_dir}/missing_regions.ply", missing_pcd)
        print(f"Saved missing regions point cloud to {output_dir}/missing_regions.ply")
    if heatmap_pcd is not None:
        o3d.io.write_point_cloud(f"{output_dir}/distance_heatmap.ply", heatmap_pcd)
        print(f"Saved distance heatmap to {output_dir}/distance_heatmap.ply")
    if regions:
        with open(f"{output_dir}/region_stats.txt", "w") as f:
            f.write(f"Total number of detected regions: {len(regions)}\n\n")
            for i, region in enumerate(regions):
                f.write(f"Region {i+1}: {len(region)} points\n")
        print(f"Saved region statistics to {output_dir}/region_stats.txt")
    with open(f"{output_dir}/change_stats.txt", "w") as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    print(f"Saved overall statistics to {output_dir}/change_stats.txt")


def pcd_to_numpy(pcd):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    return points, colors
