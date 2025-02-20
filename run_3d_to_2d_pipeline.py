import subprocess

scripts = [
            "calc_pt_cloud_normal.py", 
           "calc_curvature_roughness.py", 
           "spherical_projection.py"]

for script in scripts:
    print(f"Executing {script}...")
    subprocess.run(["python", script])
    print(f"Finished executing {script}\n")
