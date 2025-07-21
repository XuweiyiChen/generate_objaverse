#!/usr/bin/env python3
"""
Batch script to reconstruct 3D point clouds from all subfolders in the output directory.
Applies reconstruct_pointcloud.py to each subfolder that contains metadata and depth files.
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

def check_folder_has_data(folder_path):
    """Check if a folder contains the necessary files for reconstruction."""
    folder_path = Path(folder_path)
    
    # Check for metadata file
    metadata_file = folder_path / "metadata_objaverse.json"
    if not metadata_file.exists():
        return False
    
    # Check for at least one depth file
    depth_files = list(folder_path.glob("*_depth*.exr"))
    if not depth_files:
        return False
    
    # Check for at least one RGB file
    rgb_files = list(folder_path.glob("*_rgba*.png"))
    if not rgb_files:
        return False
    
    return True

def run_reconstruction(folder_path, max_depth=10.0, no_visualize=True, no_ply=False):
    """Run the reconstruction script on a single folder."""
    folder_path = Path(folder_path)
    metadata_file = folder_path / "metadata_objaverse.json"
    
    # Build command
    cmd = [
        sys.executable, "reconstruct_pointcloud.py",
        str(metadata_file),
        "--max_depth", str(max_depth)
    ]
    
    if no_visualize:
        cmd.append("--no_visualize")
    
    if no_ply:
        cmd.append("--no_ply")
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        # Run the reconstruction script
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            print(f"✅ Success: {folder_path.name}")
            if result.stdout:
                print(f"   Output: {result.stdout.strip()}")
        else:
            print(f"❌ Failed: {folder_path.name}")
            if result.stderr:
                print(f"   Error: {result.stderr.strip()}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout: {folder_path.name} (took longer than 5 minutes)")
        return False
    except Exception as e:
        print(f"💥 Exception: {folder_path.name} - {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Batch reconstruct point clouds from all output subfolders')
    parser.add_argument('--output_dir', '-o', default='/home/ubuntu/xuweiyi-us-south-3/generate_objaverse/rendering/output',
                       help='Path to the output directory containing subfolders')
    parser.add_argument('--max_depth', '-d', type=float, default=10.0,
                       help='Maximum depth to consider (meters)')
    parser.add_argument('--visualize', action='store_true',
                       help='Enable 3D visualization (disabled by default for batch processing)')
    parser.add_argument('--no_ply', action='store_true',
                       help='Skip PLY file saving')
    parser.add_argument('--start_from', type=str, default=None,
                       help='Start processing from a specific folder name')
    parser.add_argument('--max_folders', type=int, default=None,
                       help='Maximum number of folders to process')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Error: Output directory {output_dir} does not exist")
        return
    
    # Get all subfolders
    subfolders = [f for f in output_dir.iterdir() if f.is_dir()]
    subfolders.sort()  # Sort for consistent processing order
    
    print(f"Found {len(subfolders)} subfolders in {output_dir}")
    
    # Filter folders that have data
    valid_folders = []
    for folder in subfolders:
        if check_folder_has_data(folder):
            valid_folders.append(folder)
        else:
            print(f"⚠️  Skipping {folder.name} - missing required files")
    
    print(f"Found {len(valid_folders)} folders with valid data")
    
    # Apply start_from filter
    if args.start_from:
        start_idx = None
        for i, folder in enumerate(valid_folders):
            if folder.name == args.start_from:
                start_idx = i
                break
        
        if start_idx is not None:
            valid_folders = valid_folders[start_idx:]
            print(f"Starting from folder: {args.start_from}")
        else:
            print(f"Warning: Folder {args.start_from} not found, starting from beginning")
    
    # Apply max_folders filter
    if args.max_folders:
        valid_folders = valid_folders[:args.max_folders]
        print(f"Processing maximum {args.max_folders} folders")
    
    # Process each folder
    successful = 0
    failed = 0
    start_time = time.time()
    
    for i, folder in enumerate(valid_folders, 1):
        print(f"\n{'='*60}")
        print(f"Processing folder {i}/{len(valid_folders)}: {folder.name}")
        print(f"{'='*60}")
        
        # Check if pointcloud folder already exists
        pointcloud_dir = folder / "pointclouds"
        if pointcloud_dir.exists():
            print(f"⚠️  Pointcloud folder already exists: {pointcloud_dir}")
            response = input("Skip this folder? (y/n): ").lower().strip()
            if response == 'y':
                print(f"⏭️  Skipping {folder.name}")
                continue
        
        # Run reconstruction
        success = run_reconstruction(
            folder, 
            max_depth=args.max_depth,
            no_visualize=not args.visualize,
            no_ply=args.no_ply
        )
        
        if success:
            successful += 1
        else:
            failed += 1
        
        # Progress update
        elapsed = time.time() - start_time
        avg_time = elapsed / i
        remaining = avg_time * (len(valid_folders) - i)
        
        print(f"\n📊 Progress: {i}/{len(valid_folders)} ({i/len(valid_folders)*100:.1f}%)")
        print(f"⏱️  Elapsed: {elapsed/60:.1f} minutes, Estimated remaining: {remaining/60:.1f} minutes")
        print(f"✅ Successful: {successful}, ❌ Failed: {failed}")
        
        # Small delay between folders to avoid overwhelming the system
        if i < len(valid_folders):
            time.sleep(1)
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"🎉 BATCH PROCESSING COMPLETED")
    print(f"{'='*60}")
    print(f"Total folders processed: {len(valid_folders)}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"📈 Success rate: {successful/len(valid_folders)*100:.1f}%")

if __name__ == "__main__":
    main() 