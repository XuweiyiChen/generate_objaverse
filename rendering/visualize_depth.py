#!/usr/bin/env python3
"""
Script to visualize depth EXR files with different colormaps.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import OpenEXR
import Imath
import array

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

def visualize_depth(depth, output_path=None, colormap='viridis', title=None):
    """Visualize depth map with specified colormap."""
    plt.figure(figsize=(10, 8))
    
    # Create custom colormap for depth visualization
    if colormap == 'depth':
        # Custom depth colormap: blue (far) to red (near)
        colors = ['darkblue', 'blue', 'cyan', 'green', 'yellow', 'orange', 'red', 'darkred']
        cmap = LinearSegmentedColormap.from_list('depth', colors, N=256)
    else:
        cmap = plt.cm.get_cmap(colormap)
    
    # Remove infinite and very large values
    depth_clean = depth.copy()
    depth_clean[np.isinf(depth_clean)] = np.nan
    depth_clean[depth_clean > 100] = np.nan  # Remove points beyond 100 meters
    
    # Calculate meaningful depth range (exclude outliers)
    valid_depths = depth_clean[~np.isnan(depth_clean)]
    if len(valid_depths) > 0:
        depth_min = np.min(valid_depths)
        depth_max = np.percentile(valid_depths, 95)  # Use 95th percentile instead of max
    else:
        depth_min = 0
        depth_max = 1
    
    print(f"Visualization range: {depth_min:.3f}m to {depth_max:.3f}m")
    print(f"Removed {np.sum(np.isnan(depth_clean))} invalid/outlier points")
    
    im = plt.imshow(depth_clean, cmap=cmap, vmin=depth_min, vmax=depth_max)
    plt.colorbar(im, label='Depth (meters)')
    
    if title:
        plt.title(title)
    else:
        plt.title(f'Depth Map (range: {depth_min:.3f}m to {depth_max:.3f}m)')
    
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
    else:
        plt.show()
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize depth EXR files')
    parser.add_argument('depth_file', help='Path to depth EXR file')
    parser.add_argument('--output', '-o', help='Output path for visualization (PNG)')
    parser.add_argument('--colormap', '-c', default='depth', 
                       choices=['depth', 'viridis', 'plasma', 'inferno', 'magma', 'jet', 'hot', 'cool'],
                       help='Colormap for visualization')
    parser.add_argument('--title', '-t', help='Title for the plot')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.depth_file):
        print(f"Error: File {args.depth_file} does not exist")
        return
    
    try:
        # Read depth data
        print(f"Reading depth data from: {args.depth_file}")
        depth = read_exr_depth(args.depth_file)
        
        # Print depth statistics (original data)
        print(f"Original depth statistics:")
        print(f"  Min: {np.nanmin(depth):.6f} meters")
        print(f"  Max: {np.nanmax(depth):.6f} meters")
        print(f"  Mean: {np.nanmean(depth):.6f} meters")
        print(f"  Std: {np.nanstd(depth):.6f} meters")
        print(f"  Shape: {depth.shape}")
        
        # Print cleaned statistics
        depth_clean = depth.copy()
        depth_clean[np.isinf(depth_clean)] = np.nan
        depth_clean[depth_clean > 100] = np.nan
        valid_depths = depth_clean[~np.isnan(depth_clean)]
        
        if len(valid_depths) > 0:
            print(f"\nCleaned depth statistics (removing outliers):")
            print(f"  Min: {np.min(valid_depths):.6f} meters")
            print(f"  Max: {np.max(valid_depths):.6f} meters")
            print(f"  95th percentile: {np.percentile(valid_depths, 95):.6f} meters")
            print(f"  Mean: {np.mean(valid_depths):.6f} meters")
            print(f"  Std: {np.std(valid_depths):.6f} meters")
            print(f"  Valid points: {len(valid_depths)} / {depth.size}")
        else:
            print(f"\nNo valid depth points found after cleaning")
        
        # Generate output path if not provided
        if not args.output:
            base_name = os.path.splitext(args.depth_file)[0]
            args.output = f"{base_name}_visualization.png"
        
        # Visualize
        visualize_depth(depth, args.output, args.colormap, args.title)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 