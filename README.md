# Render Objaverse

## Generate shell files

```
python rendering/submit_all_generator.py \
  --obj_path ./obj_data/my_glbs \
  --save_dir ./render_output \
  --cpu_count 20 \
  --azimuth_aug 1 \
  --elevation_aug 1 \
  --frame_num 24
```

So you can submit all of them as
```
bash ./submit_all_2.sh
```

## Additional details

You can customize rendering behavior with these flags:

--mode_multi, --mode_static, --mode_front_view, etc. to control rendering styles

--two_rotate=1 to enable two-axis rotation

--azimuth_aug=1 or --elevation_aug=1 for randomized camera views

Note that Camera matrix and everything has been adjusted with respect to blender.

## Other notes

For installation and running rendering individually, please refer to ``README_old.md``.