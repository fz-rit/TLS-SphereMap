import subprocess

scripts = [
        "calc_pt_cloud_normal.py", 
        "calc_curvature_roughness.py", 
        "spherical_projection.py",
        # "data_annotation/seg_map_tools/convert_color_to_mask.py", # Only if 2D segmentation map (color) is available.
        # "data_annotation/seg_map_tools/attach_segmap_to_points.py" # Only if 2D segmentation mask is available.
        ]


for script in scripts:
    print(f"\n🔹 Executing: {script}")
    try:
        result = subprocess.run(["python", script], check=True)
        print(f"✅ Finished: {script}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {script} failed with return code {e.returncode}")
        break
else:
    print("\n🎉 All scripts executed successfully!")
