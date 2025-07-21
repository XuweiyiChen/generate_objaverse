#!/usr/bin/env python3
"""
Parallel GLB Renderer with Progress Tracking

This script renders all GLB files in parallel using Blender with tqdm progress bars.
Requires: tqdm (pip install tqdm)

Usage:
    python3 parallel_render.py --parallel_jobs 100
    python3 parallel_render.py --dry_run  # Test without executing
"""

import os
import subprocess
import sys
from pathlib import Path
import argparse

try:
    from tqdm import tqdm
except ImportError:
    print("Error: tqdm is not installed. Please install it with:")
    print("pip install tqdm")
    sys.exit(1)

def find_glb_files(base_dir):
    """Find all GLB files in the directory structure"""
    glb_files = []
    
    # First, count total directories to estimate progress
    total_dirs = sum(len(dirs) for root, dirs, files in os.walk(base_dir))
    
    with tqdm(desc="🔍 Finding GLB files", unit="dirs") as pbar:
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.glb'):
                    glb_files.append(os.path.join(root, file))
            pbar.update(1)
    
    return glb_files

def create_render_command(glb_file, blender_path, script_path, output_dir):
    """Create a render command for a single GLB file"""
    # Extract filename without extension for output directory
    filename = os.path.basename(glb_file)
    name_without_ext = os.path.splitext(filename)[0]
    
    # Create output directory path
    output_path = os.path.join(output_dir, name_without_ext)
    
    # Construct the blender command
    cmd = f'"{blender_path}" --background --python "{script_path}" -- --object_path "{glb_file}" --output_dir "{output_path}" --mode_multi 1 --frame_num 48'
    
    return cmd

def main():
    parser = argparse.ArgumentParser(description='Render all GLB files in parallel using Blender')
    parser.add_argument('--glb_dir', 
                       default='/home/ubuntu/xuweiyi-us-south-3/generate_objaverse/rendering/objaverse_curated_v2/glbs',
                       help='Directory containing GLB files')
    parser.add_argument('--blender_path', 
                       default='/home/ubuntu/xuweiyi-us-south-3/generate_objaverse/rendering/blender-3.2.2-linux-x64/blender',
                       help='Path to Blender executable')
    parser.add_argument('--script_path', 
                       default='/home/ubuntu/xuweiyi-us-south-3/generate_objaverse/rendering/blender_cpu.py',
                       help='Path to Blender rendering script')
    parser.add_argument('--output_dir', 
                       default='/home/ubuntu/xuweiyi-us-south-3/generate_objaverse/rendering/output',
                       help='Output directory for rendered images')
    parser.add_argument('--parallel_jobs', 
                       type=int, 
                       default=100,
                       help='Number of parallel jobs')
    parser.add_argument('--dry_run', 
                       action='store_true',
                       help='Show commands without executing')
    
    args = parser.parse_args()
    
    # Check if required paths exist
    if not os.path.exists(args.glb_dir):
        print(f"Error: GLB directory not found: {args.glb_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.blender_path):
        print(f"Error: Blender executable not found: {args.blender_path}")
        sys.exit(1)
    
    if not os.path.exists(args.script_path):
        print(f"Error: Blender script not found: {args.script_path}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all GLB files
    print(f"🔍 Searching for GLB files in {args.glb_dir}...")
    glb_files = find_glb_files(args.glb_dir)
    print(f"✅ Found {len(glb_files)} GLB files")
    
    if not glb_files:
        print("No GLB files found!")
        sys.exit(1)
    
    # Create commands file for GNU parallel
    commands_file = "render_commands.txt"
    
    print(f"Creating commands file: {commands_file}")
    with open(commands_file, 'w') as f:
        for glb_file in tqdm(glb_files, desc="📝 Generating commands", unit="files"):
            cmd = create_render_command(glb_file, args.blender_path, args.script_path, args.output_dir)
            f.write(cmd + '\n')
    
    if args.dry_run:
        print("Dry run mode - showing first 5 commands:")
        with open(commands_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                print(f"{i+1}: {line.strip()}")
        print(f"Total commands: {len(glb_files)}")
        return
    
    # Run commands in parallel using GNU parallel
    print(f"🚀 Starting parallel rendering with {args.parallel_jobs} jobs...")
    print(f"📊 Processing {len(glb_files)} GLB files with 48 frames each")
    print(f"🎯 Total expected outputs: {len(glb_files) * 48 * 2} files (PNG + EXR)")
    
    # Check if gnu parallel is available
    try:
        subprocess.run(['parallel', '--version'], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: GNU parallel not found. Installing...")
        try:
            subprocess.run(['sudo', 'apt-get', 'update'], check=True)
            subprocess.run(['sudo', 'apt-get', 'install', '-y', 'parallel'], check=True)
        except subprocess.CalledProcessError:
            print("Failed to install GNU parallel. Please install it manually:")
            print("sudo apt-get install parallel")
            sys.exit(1)
    
    # Run parallel processing
    parallel_cmd = [
        'parallel', 
        '-j', str(args.parallel_jobs),
        '--progress',
        '--joblog', 'parallel_jobs.log',
        '<', commands_file
    ]
    
    print(f"Running: {' '.join(parallel_cmd)}")
    
    try:
        # Use shell=True to handle the redirect properly
        result = subprocess.run(f"parallel -j {args.parallel_jobs} --progress --joblog parallel_jobs.log < {commands_file}", 
                              shell=True, 
                              check=True)
        print("✅ Parallel rendering completed successfully!")
        
        # Show summary
        if os.path.exists('parallel_jobs.log'):
            print("\n📋 Job summary:")
            subprocess.run(['tail', '-n', '10', 'parallel_jobs.log'])
            
    except subprocess.CalledProcessError as e:
        print(f"Error during parallel execution: {e}")
        sys.exit(1)
    
    # Clean up commands file
    if os.path.exists(commands_file):
        os.remove(commands_file)
    
    print(f"🎉 Rendering complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 