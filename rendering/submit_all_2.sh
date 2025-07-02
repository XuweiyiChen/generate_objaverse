#!/bin/bash

sbatch --job-name=render_task_1 --partition=standard --output=./sbatch_two_rotate_7_val_final/task_1_%J.out -A uva_cv_lab --nodes=1 --time=15:00:00 --cpus-per-task=1 ./sh_two_rotate_7_val_final/task_cpu_1.sh
