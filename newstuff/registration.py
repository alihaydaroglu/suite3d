import os, sys
sys.path.insert(0, os.getcwd())
from suite3d.reg_3d import register_2_images
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree


def voxelize_points(points: np.ndarray, voxel_size: float, padding: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts a 3D point cloud into a sparse 3D voxel grid.

    This is a critical preprocessing step for using grid-based registration
    algorithms like phase correlation on point cloud data.

    Args:
        points (np.ndarray): The input point cloud, a NumPy array of shape (N, 3),
                             where N is the number of points.
        voxel_size (float): The side length of a single cubic voxel in the
                            same units as the input points (e.g., micrometers).
                            This is the most important parameter to tune.
        padding (int): The number of empty voxels to add as a border around
                       the data. This helps prevent points from being clipped
                       at the edges of the volume.

    Returns:
        tuple[np.ndarray, np.ndarray]:
        - grid (np.ndarray): A 3D NumPy array where voxels containing one or
                             more points are set to 1 and empty voxels are 0.
        - origin (np.ndarray): A 1x3 array representing the real-world coordinate
                               (x, y, z) of the corner of the grid's first
                               voxel, grid[0, 0, 0]. This is essential for
                               converting voxel shifts back to real-world units.
    """
    if points.shape[0] == 0:
        return np.zeros((padding * 2, padding * 2, padding * 2), dtype=np.uint8), np.zeros(3)

    # 1. Find the min/max bounds of the point cloud
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)

    # 2. Define the grid's origin in the original coordinate system
    # We shift the origin by the padding to create a border.
    origin = min_coords - (padding * voxel_size)

    # 3. Calculate the required dimensions of the grid in voxels
    # Add 2 * padding to account for the border on both sides.
    span = max_coords - origin
    grid_dims = np.ceil(span / voxel_size).astype(int)

    # 4. Create the empty 3D grid
    grid = np.zeros(grid_dims, dtype=np.uint8)

    # 5. Voxelize: convert world coordinates to grid indices
    # This is the core step: shift points relative to the new origin,
    # scale by voxel size, and cast to integers.
    indices = np.floor((points - origin) / voxel_size).astype(int)

    # 6. Populate the grid
    # Use advanced NumPy indexing to set the value at the calculated indices to 1.
    # This is significantly faster than a for-loop.
    grid[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
    
    return grid, origin

def visualize_voxelization(
    points: np.ndarray,
    grid: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
    point_color: str = 'blue',
    voxel_color: str = 'red'
):
    """
    Creates a 3D visualization of a point cloud and its corresponding voxel grid.

    Args:
        points (np.ndarray): The original (N, 3) point cloud.
        grid (np.ndarray): The 3D boolean/integer grid from voxelization.
        origin (np.ndarray): The real-world coordinate of the grid's [0,0,0] corner.
        voxel_size (float): The side length of a single voxel.
        point_color (str): Matplotlib color for the original points.
        voxel_color (str): Matplotlib color for the voxel markers.
    """
    # --- 1. Set up the 3D plot ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    idx = np.random.choice(points.shape[0], size=min(1000, points.shape[0]), replace=False)
    points = points[idx]

    # --- 2. Plot the original points ---
    # We plot them as small, slightly transparent dots.
    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=point_color,
        s=10,          # Small size
        alpha=0.6,
        label='Original Points'
    )

    # --- 3. Plot the voxel representation ---
    # Find the indices of all non-zero voxels
    voxel_indices = np.argwhere(grid > 0)

    # Calculate the real-world coordinates of the CENTER of each occupied voxel
    # The `+ 0.5` is key to moving from the corner to the center.
    voxel_centers = origin + (voxel_indices + 0.5) * voxel_size
    
    marker_size = 1

    # Plot the voxel centers as large, semi-transparent squares
    ax.scatter(
        voxel_centers[:, 0], voxel_centers[:, 1], voxel_centers[:, 2],
        c=voxel_color,
        marker='s',    # Square marker
        s=marker_size,
        alpha=0.2,     # More transparent to see points inside
        label='Voxel Centers'
    )

    # --- 4. Make the plot pretty ---
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_zlabel('Z coordinate')
    ax.set_title('Point Cloud and its Voxelization')
    ax.legend()
    
    # Ensure aspect ratio is equal so voxels look like cubes
    ax.set_aspect('equal')
    
    plt.show()

def pad_grids(grid1: np.ndarray, grid2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Pads two 3D grids with zeros to make them the same shape.

    The final shape will be the element-wise maximum of the two input shapes.
    """
    shape1 = np.array(grid1.shape)
    shape2 = np.array(grid2.shape)
    
    # Determine the target shape by taking the max of each dimension
    target_shape = tuple(np.maximum(shape1, shape2))
    
    # Create new zero-filled arrays of the target shape
    padded1 = np.zeros(target_shape, dtype=grid1.dtype)
    padded2 = np.zeros(target_shape, dtype=grid2.dtype)
    
    # Copy the original grid data into the top-left corner of the new arrays
    padded1[:shape1[0], :shape1[1], :shape1[2]] = grid1
    padded2[:shape2[0], :shape2[1], :shape2[2]] = grid2
    
    return padded1, padded2

def verify_registration(
    points_ref: np.ndarray,
    points_moving: np.ndarray,
    calculated_shift: np.ndarray
):
    """
    Verifies a 3D point cloud registration result both visually and quantitatively.

    Args:
        points_ref (np.ndarray): The reference point cloud (N, 3 array).
        points_moving (np.ndarray): The moving point cloud *before* alignment (M, 3 array).
        calculated_shift (np.ndarray): The translation vector found by the registration
                                       pipeline. This is the vector that should be added
                                       to `points_moving` to align it with `points_ref`.
    
    Returns:
        dict: A dictionary containing the quantitative metrics:
              {'mean_distance_before': float, 'mean_distance_after': float}
    """
    # --- 1. Apply the calculated transformation ---
    points_moving_aligned = points_moving + calculated_shift

    # --- 2. Quantitative Verification (Mean Nearest Neighbor Distance) ---
    # Build a KD-Tree on the reference point cloud for efficient querying
    kdtree_ref = cKDTree(points_ref)

    # Query the tree to find the distance of each point in the 'moving' cloud
    # to its nearest neighbor in the 'reference' cloud.
    distances_before, _ = kdtree_ref.query(points_moving)
    distances_after, _ = kdtree_ref.query(points_moving_aligned)

    mean_dist_before = np.mean(distances_before)
    mean_dist_after = np.mean(distances_after)

    print("\n--- Quantitative Verification ---")
    print(f"Mean Nearest Neighbor Distance BEFORE alignment: {mean_dist_before:.4f}")
    print(f"Mean Nearest Neighbor Distance AFTER alignment:  {mean_dist_after:.4f}")
    improvement = (mean_dist_before - mean_dist_after) / mean_dist_before * 100
    print(f"Improvement: {improvement:.2f}%")
    
    # --- 3. Visual Verification (Side-by-side 3D plot) ---
    fig = plt.figure(figsize=(16, 8))

    idx = np.random.choice(min(points_ref.shape[0], points_moving.shape[0]), size=min(900, points_ref.shape[0]), replace=False)
    points_ref = points_ref[idx]
    points_moving = points_moving[idx]
    points_moving_aligned = points_moving_aligned[idx]
    
    # --- Left Plot: Before Alignment ---
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(points_ref[:, 0], points_ref[:, 1], points_ref[:, 2], 
                c='red', s=5, alpha=0.5, label='Reference Cloud')
    ax1.scatter(points_moving[:, 0], points_moving[:, 1], points_moving[:, 2], 
                c='blue', s=5, alpha=0.5, label='Moving Cloud (Original)')
    ax1.set_title('Before Alignment')
    ax1.set_xlabel('Z'); ax1.set_ylabel('Y'); ax1.set_zlabel('X')
    ax1.legend()
    ax1.set_aspect('equal')

    # --- Right Plot: After Alignment ---
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(points_ref[:, 0], points_ref[:, 1], points_ref[:, 2], 
                c='red', s=5, alpha=0.5, label='Reference Cloud')
    ax2.scatter(points_moving_aligned[:, 0], points_moving_aligned[:, 1], points_moving_aligned[:, 2], 
                c='blue', s=5, alpha=0.5, label='Moving Cloud (Aligned)')
    ax2.set_title('After Alignment')
    ax2.set_xlabel('Z'); ax2.set_ylabel('Y'); ax2.set_zlabel('X')
    ax2.legend()
    ax2.set_aspect('equal')
    
    plt.suptitle('Registration Verification', fontsize=16)
    plt.show()

    return {
        'mean_distance_before': mean_dist_before,
        'mean_distance_after': mean_dist_after
    }


session1 = r"\\znas.cortexlab.net\Lab\Share\Ali\for-suyash\registration_test\s3d-results-AH012_2024-08-06_2-3-4-5-6-7-8\stats.npy"
session2 = r"\\znas.cortexlab.net\Lab\Share\Ali\for-suyash\registration_test\s3d-results-AH012_2024-08-09_0-1-2-3-4-5-6-7-8-9-10-11-12-13-14\stats.npy"

stat1 = np.load(session1, allow_pickle=True)
stat2 = np.load(session2, allow_pickle=True)

vol1, vol2 = [], []

for cell in stat1:
    z,y,x = cell['coords']
    mid = [np.mean(z), np.mean(y), np.mean(x)]
    vol1.append(mid)
for cell in stat2:
    z,y,x = cell['coords']
    mid = [np.mean(z), np.mean(y), np.mean(x)]
    vol2.append(mid)

vol1 = np.array(vol1)
vol2 = np.array(vol2)

center1 = np.mean(vol1, axis=0)
center2 = np.mean(vol2, axis=0)
print(center1, center2)
print("Initial shift (world coords): ", center2 - center1)

VOXEL_SIZE = 2

grid1, origin1 = voxelize_points(vol1, voxel_size=VOXEL_SIZE)
grid2, origin2 = voxelize_points(vol2, voxel_size=VOXEL_SIZE)

grid1, grid2 = pad_grids(grid1, grid2)

pc, shift_in_voxels, _ ,_ = register_2_images(grid1, grid2, np.array([3, 3, 3]))
print("Calculated shift (voxels): ", shift_in_voxels)
print("Origin1: ", origin1)
print("Origin2: ", origin2)
print("Origin diff (world coords): ", origin1 - origin2)

total_shift_in_world_coords = (shift_in_voxels * VOXEL_SIZE) + (origin1 - origin2)
print("Calculated shift (world coords): ", total_shift_in_world_coords)

result = verify_registration(points_ref=vol1, points_moving=vol2, calculated_shift=total_shift_in_world_coords)
print(result)

# visualize_voxelization(vol1, grid1, origin1, voxel_size=5)

