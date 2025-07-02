import os
import shutil

def force_remove_readonly(func, path, exc_info):
    """Change the file to writable and retry removal"""
    os.chmod(path, 0o777)  # Change permission to allow deletion
    func(path)  # Retry removing

base_path = "/project/uva_cv_lab/xuweic/Diffusion4D/rendering/final-final-marstin-fov50"

for subdir in os.listdir(base_path):
    subdir_path = os.path.join(base_path, subdir)
    
    # Ensure it's a directory
    if os.path.isdir(subdir_path):
        # Check if there's any .json file in the directory
        has_json = any(file.endswith(".json") for file in os.listdir(subdir_path))
        
        # If no JSON file is found, delete the directory
        if not has_json:
            try:
                shutil.rmtree(subdir_path, onerror=force_remove_readonly)
                print(f"Deleted: {subdir_path}")
            except Exception as e:
                print(f"Failed to delete {subdir_path}: {e}")