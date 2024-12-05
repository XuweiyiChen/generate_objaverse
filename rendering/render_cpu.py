import os
import subprocess
import logging
import argparse
from math import ceil
import random

parser = argparse.ArgumentParser()
parser.add_argument("--obj_path", type=str, default="./obj_data/hf-objaverse-v1/glbs")
parser.add_argument("--save_dir", type=str, default="./output")
parser.add_argument("--cpu_count", type=int, default=10, help="Number of CPUs available for processing")
parser.add_argument("--frame_num", type=int, default=24)
parser.add_argument("--azimuth_aug", type=int, default=0)
parser.add_argument("--elevation_aug", type=int, default=0)
parser.add_argument("--resolution", default=256)
parser.add_argument("--mode_multi", type=int, default=0)
parser.add_argument("--mode_static", type=int, default=0)
parser.add_argument("--mode_front_view", type=int, default=0)
parser.add_argument("--mode_four_view", type=int, default=0)
parser.add_argument("--sh_dir", type=str, default="./sh_scripts", help="Directory to save generated .sh files")
parser.add_argument("--sbatch_dir", type=str, default="./sbatch_logs", help="Directory to save sbatch logs")
parser.add_argument("--sbatch_save_file", type=str, default="./submit_all.sh", help="File to save all sbatch commands")

args = parser.parse_args()

# Create necessary directories
os.makedirs(args.save_dir, exist_ok=True)
os.makedirs(args.sh_dir, exist_ok=True)
os.makedirs(args.sbatch_dir, exist_ok=True)

# Read objaverse GLB paths
glb_files = []
for root, dirs, files in os.walk(args.obj_path):
    for file in files:
        if file.endswith(".glb"):
            glb_file_path = os.path.join(root, file)
            glb_files.append(glb_file_path)

# Split tasks among CPUs
num_tasks = len(glb_files)
tasks_per_cpu = ceil(num_tasks / args.cpu_count)
cpu_task_groups = [
    glb_files[i * tasks_per_cpu:(i + 1) * tasks_per_cpu]
    for i in range(args.cpu_count)
]


def generate_command(args, glb_file, file_name):
    """Generate the Blender command for a single object."""
    if args.azimuth_aug:
        azimuth = round(random.uniform(0, 1), 2)
        file_name += f"_az{azimuth:.2f}"
    else:
        azimuth = 0

    if args.elevation_aug:
        elevation = random.randint(5, 30)
        file_name += f"_el{elevation:.2f}"
    else:
        elevation = 0

    save_path = os.path.join(args.save_dir, file_name)
    os.makedirs(save_path, exist_ok=True)  # Ensure each save directory is created
    command = f"blender-3.2.2-linux-x64/blender \
        --background --python blender_cpu.py -- \
        --object_path {glb_file} \
        --frame_num {args.frame_num} \
        --output_dir {save_path} \
        --azimuth {azimuth} \
        --elevation {elevation} \
        --resolution {args.resolution} \
        --mode_multi {args.mode_multi} \
        --mode_static {args.mode_static} \
        --mode_front {args.mode_front_view} \
        --mode_four_view {args.mode_four_view}"
    return command


# Generate .sh scripts for each CPU's tasks
sh_files = []
sbatch_commands = []
logging.basicConfig(level=logging.INFO)

for cpu_idx, task_group in enumerate(cpu_task_groups):
    sh_file = os.path.join(args.sh_dir, f"task_cpu_{cpu_idx + 1}.sh")
    sh_files.append(sh_file)
    with open(sh_file, "w") as f:
        f.write("#!/bin/bash\n\n")
        for glb_file in task_group:
            file_name = os.path.basename(glb_file).split(".")[0]
            command = generate_command(args, glb_file, file_name)
            f.write(command + "\n")
    os.chmod(sh_file, 0o755)
    logging.info(f"Generated script: {sh_file}")

    # Generate sbatch command for this .sh file
    sbatch_command = (
        f"sbatch --job-name=render_task_{cpu_idx + 1} --partition=standard "
        f"--output={os.path.join(args.sbatch_dir, f'task_{cpu_idx + 1}_%J.out')} "
        f"-A uva_cv_lab --nodes=1 --time=2:00:00 --cpus-per-task=1 {sh_file}"
    )
    sbatch_commands.append(sbatch_command)

# Save all sbatch commands to a single file
with open(args.sbatch_save_file, "w") as f:
    f.write("#!/bin/bash\n\n")
    for sbatch_command in sbatch_commands:
        f.write(sbatch_command + "\n")
os.chmod(args.sbatch_save_file, 0o755)

logging.info(f"All sbatch commands saved in {args.sbatch_save_file}")
print(f"All sbatch commands saved in {args.sbatch_save_file}")