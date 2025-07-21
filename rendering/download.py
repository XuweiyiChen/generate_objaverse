import argparse
import objaverse
import multiprocessing
import os
import time
import random

parser = argparse.ArgumentParser()
parser.add_argument("--id_path", type=str, default="src/missing.txt")
args = parser.parse_args()

# Use very conservative process count to avoid rate limiting
max_processes = 1  # Start with single process to avoid rate limiting

with open(args.id_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()
object_uids = [line.strip() for line in lines]
print('file number:', len(object_uids))

# objaverse.BASE_PATH = '/home/projects/Diffusion4D/rendering/'
objaverse._VERSIONED_PATH = 'objaverse_curated_v2/'

def download_with_retry(uids, processes=1, max_retries=3, base_delay=60):
    """Download objects with exponential backoff retry logic."""
    for attempt in range(max_retries):
        try:
            print(f"Download attempt {attempt + 1}/{max_retries} with {processes} processes...")
            
            # Add initial delay to avoid immediate rate limiting
            if attempt > 0:
                delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 10)
                print(f"Waiting {delay:.1f} seconds before retry...")
                time.sleep(delay)
            
            objaverse.load_objects(
                uids=uids,
                download_processes=processes
            )
            print('Download finished successfully!')
            return True
            
        except Exception as e:
            error_msg = str(e)
            print(f"Attempt {attempt + 1} failed: {error_msg}")
            
            if "429" in error_msg or "Too Many Requests" in error_msg:
                if attempt < max_retries - 1:
                    print("Rate limited - will retry with longer delay...")
                    continue
                else:
                    print("Max retries exceeded for rate limiting")
                    return False
            else:
                print(f"Non-rate-limit error: {error_msg}")
                return False
    
    return False

# Try different strategies with rate limiting protection
success = False

# Strategy 1: Single process with retry
print("=== Strategy 1: Single process with retry ===")
success = download_with_retry(object_uids, processes=1, max_retries=3, base_delay=60)

if not success:
    print("\n=== Strategy 2: Smaller batches ===")
    # Strategy 2: Download in smaller batches
    batch_size = min(10, len(object_uids))  # Process 10 objects at a time
    total_batches = (len(object_uids) + batch_size - 1) // batch_size
    
    for i in range(0, len(object_uids), batch_size):
        batch_uids = object_uids[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        print(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch_uids)} objects)")
        
        batch_success = download_with_retry(batch_uids, processes=1, max_retries=2, base_delay=30)
        
        if batch_success:
            print(f"Batch {batch_num} completed successfully")
            # Add delay between batches to avoid rate limiting
            if i + batch_size < len(object_uids):
                delay = random.uniform(5, 15)
                print(f"Waiting {delay:.1f} seconds before next batch...")
                time.sleep(delay)
        else:
            print(f"Batch {batch_num} failed - continuing with next batch")

print("\nDownload process completed. Check individual batch results above.")