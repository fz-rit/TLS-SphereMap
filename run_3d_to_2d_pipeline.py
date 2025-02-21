import subprocess

scripts = [
            "calc_pt_cloud_normal.py", 
           "calc_curvature_roughness.py", 
           "spherical_projection.py"
        ]

for script in scripts:
    print(f"Executing {script}...")
    process = subprocess.run(["python", script])
    print(f"Finished executing {script}\n")
    if process.returncode != 0:
        print(f"Error: {script} failed with return code {process.returncode}")
        break
print("All scripts executed successfully!")
