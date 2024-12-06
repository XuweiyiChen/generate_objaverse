# Example: Compressing 100 subdirectories at a time
base_path="/project/uva_cv_lab/xuweic/Diffusion4D/rendering/output"
output_path="/project/uva_cv_lab/xuweic/4D-objaverse-1.0"
mkdir -p "$output_path"

find "$base_path" -mindepth 1 -maxdepth 1 -type d | split -l 100 - temp_dirs

# Compress batches in parallel
n=1
for list in temp_dirs*; do
    (
        zip -r "$output_path/batch_$n.zip" $(cat $list) &
    )
    n=$((n + 1))
done
wait
rm temp_dirs*