import subprocess

scripts = [
        "processing.calc_pt_cloud_normal", 
        # "processing.calc_curvature_roughness", # To be revised for calculating eigen-based scalar fields like curvature, roughness, and so on.
        # "processing.spherical_projection",
        # "processing.spherical_back_projection",
        # "data_annotation.seg_map_tools.convert_color_to_mask", # Only if 2D segmentation map (color) is available.
        # "data_annotation.seg_map_tools.attach_segmap_to_points" # Only if 2D segmentation mask is available.
        ]


for script in scripts:
    print(f"\n🔹 Executing: {script}")
    try:
        result = subprocess.run(["python", "-m", script], check=True)
        print(f"✅ Finished: {script}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {script} failed with return code {e.returncode}")
        break
else:
    print("\n🎉 All scripts executed successfully!")
