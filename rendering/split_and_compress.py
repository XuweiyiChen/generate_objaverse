# Usage: python split_and_compress.py --source_dir /path/to/source/folder --output_dir /path/to/output/folder --max_size_mb 100 --workers 4
import os
import time
import zipfile
import argparse
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, Manager, Value, Array


def scan_directory_worker(to_be_scanned_queue, file_list, worker_id, is_worker_active, all_done, subfolders_to_include=None, subdir_parallelization_threshold=3):
    # Initialize a progress bar for this worker
    progress_bar = tqdm(desc=f"Worker {worker_id + 1}", position=worker_id, leave=True)

    while True:
        try:
            # Wait indefinitely for a directory to process until the all_done flag is set and the queue is empty
            directory = to_be_scanned_queue.get(timeout=1.0)
            is_worker_active[worker_id] = 1  # Mark the worker as active
        except:
            # If timeout and all_done is set and queue is empty, we exit the loop
            is_worker_active[worker_id] = 0  # Mark the worker as idle
            if all_done.value and to_be_scanned_queue.qsize() == 0:
                break
            continue

        # Process the directory
        file_count = 0
        for root, dirs, files in os.walk(directory, followlinks=False):
            subdirs_to_add = []

            # Skip subdirectories not in our include list if provided
            if subfolders_to_include is not None:
                # Get the top-level subdirectory under the source root
                rel_path = os.path.relpath(root, directory)
                if rel_path != '.':  # Not the source_dir itself
                    top_level_dir = rel_path.split(os.sep)[0]
                    if top_level_dir not in subfolders_to_include:
                        dirs.clear()  # Skip all subdirectories
                        continue      # Skip current directory

            # Process files in the current directory
            for file in files:
                file_path = os.path.join(root, file)

                if os.path.islink(file_path):
                    # Record symlink with size 0 (since symlinks don't have a file size in this case)
                    file_list.append((file_path, 0))
                else:
                    # Record regular file
                    file_size = os.path.getsize(file_path)
                    file_list.append((file_path, file_size))

                file_count += 1

            # If the directory has more than 10 subdirectories, add them to the queue
            if len(dirs) > subdir_parallelization_threshold:
                for subdir in dirs:
                    subdir_path = os.path.join(root, subdir)
                    subdirs_to_add.append(subdir_path)

                # Stop descending into subdirectories, add them to the queue
                dirs.clear()

            # Add subdirectories to the queue if there are more than 10
            for subdir in subdirs_to_add:
                to_be_scanned_queue.put(subdir)

            # Update progress bar for this worker
            progress_bar.update(file_count)

    # Close the progress bar after the worker is done
    progress_bar.close()

def parallel_scan(source_dir, num_workers, subfolders_to_include=None):
    file_list = Manager().list()  # this need to be put outside the manager context so that it can be accessed outside the manager context
    with Manager() as manager:
        to_be_scanned_queue = manager.Queue()
        all_done = manager.Value('i', 0)  # Shared integer to indicate all_done status
        is_worker_active = manager.Array('i', [1] * num_workers)  # Track worker activity

        # Initially add the source directory to the queue
        to_be_scanned_queue.put(source_dir)

        with Pool(processes=num_workers) as pool:
            results = [pool.apply_async(scan_directory_worker, (to_be_scanned_queue, file_list, i, is_worker_active, all_done, subfolders_to_include)) for i in range(num_workers)]

            # Monitor worker activity to know when to terminate
            while True:
                time.sleep(0.5)  # Prevent tight loop
                # Check if all workers are idle and queue is empty
                if all(is_worker_active[i] == 0 for i in range(num_workers)) and to_be_scanned_queue.qsize() == 0:
                    print("All workers are idle and queue is empty. Scan has finished.")
                    break  # Exit the loop if all workers are idle and the queue is empty

            # Set all_done flag when all work is presumed to be completed
            for r in results:
                r.wait()
            all_done.value = 1

    return list(file_list)


# Function to read subfolders list from file
def read_subfolders_list(file_path):
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} does not exist. Will process all subfolders.")
        return None
    
    with open(file_path, 'r') as f:
        # Read all lines and strip whitespace
        subfolders = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"Loaded {len(subfolders)} subfolders to include from {file_path}")
    return subfolders


# Function to form chunks from the scanned files
def form_chunks(file_list, max_size_bytes):
    chunk_list = []
    current_chunk_size = 0
    current_files = []

    for file_path, file_size in file_list:
        if current_chunk_size + file_size > max_size_bytes and current_files:
            chunk_list.append(current_files)
            current_files = []
            current_chunk_size = 0

        current_files.append(file_path)
        current_chunk_size += file_size

    if current_files:
        chunk_list.append(current_files)

    return chunk_list


# Function to compress a single chunk with progress bar
def compress_chunk(i, chunk_list, output_dir, folder_name, source_dir, progress_queue, num_workers):
    zip_name = os.path.join(output_dir, f'{folder_name}_{i + 1}.zip')

    # Create the progress bar for this process
    progress_bar = tqdm(total=len(chunk_list[i]), desc=f"{folder_name}_{i + 1}.zip", position=i, leave=True)

    # Compress the chunk and update progress
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in chunk_list[i]:
            arcname = os.path.relpath(file, start=source_dir)
            if os.path.islink(file):
                # Read the symlink target
                symlink_target = os.readlink(file)

                # Convert absolute symlink to relative if necessary
                if os.path.isabs(symlink_target):
                    symlink_target = os.path.relpath(symlink_target, start=os.path.dirname(file))

                # Create a ZipInfo object for the symlink
                info = zipfile.ZipInfo(arcname)
                info.create_system = 3  # Unix system
                info.external_attr = 0xA1FF0000  # File type and permissions for symlink

                # Add the relative symlink target as the file content
                zipf.writestr(info, symlink_target)
            else:
                zipf.write(file, arcname)

            # Update progress for this file
            progress_bar.update(1)

    progress_bar.close()

    # Notify that this chunk is done
    progress_queue.put(1)  # Send an update to the queue


# Parallel compression with separate progress bars using multiprocessing
def compress_all_chunks(chunk_list, output_dir, folder_name, source_dir, num_workers=4):
    total_chunks = len(chunk_list)

    # Use Manager to share progress data across processes
    with Manager() as manager:
        progress_queue = manager.Queue()  # Queue for tracking finished chunks

        # Start multiprocessing pool
        with Pool(processes=num_workers) as pool:
            # Launch the pool of worker processes
            pool.starmap_async(
                compress_chunk,
                [(i, chunk_list, output_dir, folder_name, source_dir, progress_queue, num_workers) for i in range(total_chunks)]
            )

            # Wait for all processes to finish
            pool.close()
            pool.join()

def main():
    parser = argparse.ArgumentParser(description="Split and compress a folder into chunks.")

    # Named arguments for better usability
    parser.add_argument("--source_dir", required=True, help="The source directory to split and compress.")
    parser.add_argument("--output_dir", required=True, help="The output directory for the compressed chunks.")
    parser.add_argument("--max_size_mb", type=int, required=True, help="Maximum size of each chunk in MB.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers for compression.")
    parser.add_argument("--include_file", type=str, help="Path to a file containing a list of subfolders to include.")

    args = parser.parse_args()

    source_dir = args.source_dir
    output_dir = args.output_dir
    max_size_mb = args.max_size_mb
    num_workers = args.workers
    include_file = args.include_file

    # Get the folder name (basename) of the source directory
    folder_name = os.path.basename(os.path.normpath(source_dir))

    # Ensure the output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Convert MB to bytes
    max_size_bytes = max_size_mb * 1024 * 1024

    # Read the list of subfolders to include if specified
    subfolders_to_include = None
    if include_file:
        subfolders_to_include = read_subfolders_list(include_file)

    # Stage 1: Parallel directory scan
    # time the parallel scan
    start_time = time.time()
    print("Starting the dynamic parallel directory scan...")
    file_list = parallel_scan(source_dir, num_workers, subfolders_to_include)
    print(f"Finished scanning. {len(file_list)} files were found.")
    print(f"Scanning took {time.time() - start_time:.2f} seconds")

    # Stage 2: Form chunks after all files have been scanned
    print("Forming chunks...")
    chunk_list = form_chunks(file_list, max_size_bytes)
    print(f"{len(chunk_list)} chunks were created.")

    # Stage 3: Compress each chunk in parallel with separate progress bars
    print("Starting compression...")
    compress_all_chunks(chunk_list, output_dir, folder_name, source_dir, num_workers=num_workers)
    print("Compression completed.")


if __name__ == "__main__":
    main()
