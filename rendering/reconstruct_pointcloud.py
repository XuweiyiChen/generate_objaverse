#!/usr/bin/env python3
"""
Script to reconstruct 3D point clouds from depth images and camera poses.
Uses the metadata from Blender rendering to convert depth maps to 3D points.
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import OpenEXR
import Imath
import array
from PIL import Image

def read_exr_depth(filepath):
    """Read depth values from EXR file."""
    if not OpenEXR.isOpenExrFile(filepath):
        raise ValueError(f"File {filepath} is not an EXR file")
    
    file = OpenEXR.InputFile(filepath)
    dw = file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    
    # Read the depth channel (usually stored in R channel for RGB EXR)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    depth_str = file.channel("R", FLOAT)
    depth = array.array('f', depth_str).tolist()
    depth = np.array(depth).reshape(size[1], size[0])
    
    file.close()
    return depth

def read_rgb_image(filepath):
    """Read RGB image and return as numpy array."""
    img = Image.open(filepath)
    return np.array(img) / 255.0  # Normalize to [0, 1]

def depth_to_pointcloud(depth, rgb=None, fx=274.5, fy=274.5, cx=128.0, cy=128.0, 
                       w2c_matrix=None, max_depth=100.0):
    """
    Convert depth map to 3D point cloud.
    
    Args:
        depth: Depth map (H, W)
        rgb: RGB image (H, W, 3) or None
        fx, fy: Focal lengths
        cx, cy: Principal point
        w2c_matrix: World to camera transformation matrix (4x4)
        max_depth: Maximum depth to consider (remove outliers)
    
    Returns:
        points_3d: Nx3 array of 3D points in world coordinates
        colors: Nx3 array of colors (if rgb provided)
    """
    h, w = depth.shape
    
    # Create pixel coordinates
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    # Remove invalid depth values
    valid_mask = (depth > 0) & (depth < max_depth) & ~np.isinf(depth)
    
    if not np.any(valid_mask):
        return np.array([]), np.array([])
    
    # Get valid pixel coordinates and depths
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    depth_valid = depth[valid_mask]
    
    # Convert to camera coordinates
    # Z = depth
    # X = (u - cx) * Z / fx
    # Y = (v - cy) * Z / fy
    z_cam = depth_valid
    x_cam = (x_valid - cx) * z_cam / fx
    y_cam = (y_valid - cy) * z_cam / fy
    
    # Stack into homogeneous coordinates (N, 4)
    points_cam = np.column_stack([x_cam, y_cam, z_cam, np.ones_like(z_cam)])
    
    # Transform to world coordinates
    if w2c_matrix is not None:
        w2c = np.array(w2c_matrix)
        c2w = np.linalg.inv(w2c)  # Camera to world transformation
        points_world = (c2w @ points_cam.T).T
        points_3d = points_world[:, :3]  # Remove homogeneous coordinate
    else:
        points_3d = points_cam[:, :3]
    
    # Get colors if RGB image provided
    colors = None
    if rgb is not None:
        colors = rgb[valid_mask]
    
    return points_3d, colors

def load_metadata(metadata_path):
    """Load metadata from JSON file."""
    with open(metadata_path, 'r') as f:
        return json.load(f)

def reconstruct_from_metadata(metadata_path, output_dir=None, max_depth=10.0, 
                            visualize=True, save_ply=True):
    """
    Reconstruct 3D point clouds from metadata and depth images.
    
    Args:
        metadata_path: Path to metadata JSON file
        output_dir: Output directory for point clouds
        max_depth: Maximum depth to consider
        visualize: Whether to create 3D visualization
        save_ply: Whether to save PLY files
    """
    # Load metadata
    metadata = load_metadata(metadata_path)
    
    # Create output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(metadata_path), "pointclouds")
    os.makedirs(output_dir, exist_ok=True)
    
    all_points = []
    all_colors = []
    frame_info = []
    
    # Process each frame
    for frame_idx, frame_data in enumerate(metadata['frames']):
        for view_name, view_data in frame_data.items():
            if not view_data:  # Skip empty views
                continue
                
            print(f"Processing frame {frame_idx}, view: {view_name}")
            
            # Get file paths
            rgb_path = view_data['file_path']
            # Handle different naming patterns
            if '_rgba.png' in rgb_path:
                depth_path = rgb_path.replace('_rgba.png', '_depth.exr')
            elif '_rgba' in rgb_path and '.png' in rgb_path:
                # Handle pattern like multi_frame24_rgba0024.png -> multi_frame24_depth0024.exr
                base_name = rgb_path.split('_rgba')[0]
                frame_suffix = rgb_path.split('_rgba')[1].replace('.png', '')
                depth_path = f"{base_name}_depth{frame_suffix}.exr"
            else:
                # Fallback: try to construct depth path
                base_name = rgb_path.replace('.png', '')
                depth_path = f"{base_name.replace('_rgba', '_depth')}.exr"
            
            # Check if files exist
            if not os.path.exists(depth_path):
                print(f"  Depth file not found: {depth_path}")
                continue
            
            # Read depth and RGB
            try:
                depth = read_exr_depth(depth_path)
                rgb = read_rgb_image(rgb_path)
                
                # Get camera parameters
                fx = view_data['fx']
                fy = view_data['fy']
                cx = view_data['cx']
                cy = view_data['cy']
                w2c = view_data['w2c']
                
                # Convert to point cloud
                points_3d, colors = depth_to_pointcloud(
                    depth, rgb, fx, fy, cx, cy, w2c, max_depth
                )
                
                if len(points_3d) > 0:
                    all_points.append(points_3d)
                    if colors is not None:
                        all_colors.append(colors)
                    
                    frame_info.append({
                        'frame': frame_idx,
                        'view': view_name,
                        'timestamp': view_data.get('timestamp', frame_idx),
                        'num_points': len(points_3d),
                        'camera_location': view_data.get('blender_camera_location', [0, 0, 0])
                    })
                    
                    print(f"  Generated {len(points_3d)} points")
                    
                    # Save individual frame PLY
                    if save_ply:
                        ply_path = os.path.join(output_dir, f"frame_{frame_idx}_{view_name}.ply")
                        save_ply_file(ply_path, points_3d, colors)
                        print(f"  Saved PLY: {ply_path}")
                else:
                    print(f"  No valid points generated")
                    
            except Exception as e:
                print(f"  Error processing frame: {e}")
    
    # Combine all point clouds
    if all_points:
        combined_points = np.vstack(all_points)
        combined_colors = np.vstack(all_colors) if all_colors else None
        
        print(f"\nCombined point cloud: {len(combined_points)} points")
        
        # Save combined PLY
        if save_ply:
            combined_ply_path = os.path.join(output_dir, "combined_pointcloud.ply")
            save_ply_file(combined_ply_path, combined_points, combined_colors)
            print(f"Saved combined PLY: {combined_ply_path}")
        
        # Save frame info
        info_path = os.path.join(output_dir, "frame_info.json")
        with open(info_path, 'w') as f:
            json.dump(frame_info, f, indent=2)
        print(f"Saved frame info: {info_path}")
        
        # Visualize
        if visualize:
            visualize_pointcloud(combined_points, combined_colors, output_dir)
        
        return combined_points, combined_colors, frame_info
    else:
        print("No valid point clouds generated")
        return None, None, []

def save_ply_file(filepath, points, colors=None):
    """Save point cloud as PLY file."""
    with open(filepath, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Data
        for i in range(len(points)):
            x, y, z = points[i]
            if colors is not None:
                r, g, b = colors[i][:3]  # Take RGB, ignore alpha
                f.write(f"{x} {y} {z} {int(r*255)} {int(g*255)} {int(b*255)}\n")
            else:
                f.write(f"{x} {y} {z}\n")

def visualize_pointcloud(points, colors=None, output_dir=None):
    """Create 3D visualization of point cloud."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Sample points for visualization (if too many)
    if len(points) > 10000:
        indices = np.random.choice(len(points), 10000, replace=False)
        points_viz = points[indices]
        colors_viz = colors[indices] if colors is not None else None
    else:
        points_viz = points
        colors_viz = colors
    
    # Plot points
    if colors_viz is not None:
        ax.scatter(points_viz[:, 0], points_viz[:, 1], points_viz[:, 2], 
                  c=colors_viz, s=0.1, alpha=0.8)
    else:
        ax.scatter(points_viz[:, 0], points_viz[:, 1], points_viz[:, 2], 
                  s=0.1, alpha=0.8)
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title(f'3D Point Cloud ({len(points)} points)')
    
    # Equal aspect ratio
    max_range = np.array([points_viz[:, 0].max() - points_viz[:, 0].min(),
                          points_viz[:, 1].max() - points_viz[:, 1].min(),
                          points_viz[:, 2].max() - points_viz[:, 2].min()]).max() / 2.0
    
    mid_x = (points_viz[:, 0].max() + points_viz[:, 0].min()) * 0.5
    mid_y = (points_viz[:, 1].max() + points_viz[:, 1].min()) * 0.5
    mid_z = (points_viz[:, 2].max() + points_viz[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if output_dir:
        viz_path = os.path.join(output_dir, "pointcloud_visualization.png")
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization: {viz_path}")
    else:
        plt.show()
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Reconstruct 3D point clouds from depth images and camera poses')
    parser.add_argument('metadata_path', help='Path to metadata JSON file')
    parser.add_argument('--output_dir', '-o', help='Output directory for point clouds')
    parser.add_argument('--max_depth', '-d', type=float, default=10.0, 
                       help='Maximum depth to consider (meters)')
    parser.add_argument('--no_visualize', action='store_true', 
                       help='Skip 3D visualization')
    parser.add_argument('--no_ply', action='store_true', 
                       help='Skip PLY file saving')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.metadata_path):
        print(f"Error: Metadata file {args.metadata_path} does not exist")
        return
    
    try:
        reconstruct_from_metadata(
            args.metadata_path,
            args.output_dir,
            args.max_depth,
            visualize=not args.no_visualize,
            save_ply=not args.no_ply
        )
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 