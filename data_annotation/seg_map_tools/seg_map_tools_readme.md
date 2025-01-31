# Segmentation Map Tools

This folder contains tools for managing **semantic segmentation datasets** in the **PASCAL VOC format**, including:
- Converting **colorful segmentation maps** to **grayscale class index masks**.
- Visualizing class index masks.
- Managing dataset-specific label maps.
- Attaching segmentation class labels to a **3D point cloud**.

---

## 📁 Files in This Folder
| File | Description |
|------|------------|
| `convert_color_to_mask.py` | Converts a **colorful segmentation map** to a **grayscale class index mask**. |
| `visualize_mask.py` | Applies a **colormap (`jet`)** to a grayscale class index mask for visualization. |
| `attach_segmap_to_points.py` | Attaches class IDs from a segmentation map and corresponding colors to a 3D point cloud, saving it as a `.csv` file. |
| `label_maps.json` | Stores **dataset-specific colormaps and class names**. |
| `seg_map_tools_readme.md` | This documentation file. |

---

## 🔧 Example Workflow
```bash
# Step 1: Convert a colorful segmentation map to grayscale class index mask
python convert_color_to_mask.py -i ../examples/seg_map_harvard_33_02.png -o ../examples/seg_map_harvard_33_02_mask.png -d HARVARD_FOREST

# [Optional] Step 2: Visualize the generated class index mask
python visualize_mask.py -i ../examples/seg_map_harvard_33_02_mask.png

# Step 3: Attach segmentation labels to a 3D point cloud
python attach_segmap_to_points.py -pf ../../input_params/attach_segmap_to_points_zmachine.json
```
✔ **Generated Files:**
- `seg_map_harvard_33_02_mask.png` → Grayscale class index mask.
- `visualized_mask.png` → Colorized visualization of the mask.
- `point_cloud_with_segmentation.csv` → Point cloud with attached segmentation labels.

---

## 📚 Detailed Documentation

### 🔄 Converting a Colorful Segmentation Map to a Class Index Mask

Use **`convert_color_to_mask.py`** to **convert a color-segmented `.png` file into a grayscale class index mask**.

#### **Usage:**
```bash
python convert_color_to_mask.py -i path/to/colorful_map.png -o path/to/output_mask.png -d MANGROVE_ROOTS
```

#### **Arguments:**
- `-i` / `--input` → Path to the input **colorful segmentation map**.
- `-o` / `--output` → Path to save the **output grayscale class index mask**.
- `-d` / `--dataset` → Dataset name from `label_maps.json` (e.g., `MANGROVE_ROOTS` or `HARVARD_FOREST`).

Example:
```bash
python convert_color_to_mask.py -i sample_map.png -o sample_mask.png -d HARVARD_FOREST
```
✔ **Output:** A grayscale `.png` where pixel values correspond to class indices.

---

### 🎨 Visualizing a Class Index Mask

Use **`visualize_mask.py`** to **apply a colormap (`jet`)** and visualize a class index mask.

#### **Usage:**
```bash
python visualize_mask.py -i path/to/class_index_mask.png
```
or save the visualization:
```bash
python visualize_mask.py -i path/to/class_index_mask.png -s
```

#### **Arguments:**
- `-i` / `--input` → Path to the class index mask (`.png`).
- `-s` / `--save` (optional) → Saves the visualization as `visualized_mask.png`.

✔ **Output:** Displays the class index mask with colors for better visibility.

---

### 📌 Attaching Segmentation Labels to a 3D Point Cloud

Use **`attach_segmap_to_points.py`** to **attach class labels from a segmentation mask to a point cloud file**.

#### **Usage:**
```bash
python attach_segmap_to_points.py -pf path/to/params.json
```

#### **Example Command:**
```bash
python attach_segmap_to_points.py -pf ../../input_params/attach_segmap_to_points_zmachine.json
```

#### **JSON Parameter File (`params.json`):**
```json
{
    "root_dir": "D:/mylab/tls_point_segmentation",
    "pointcloud": "data/point_cloud.txt",
    "segmap": "data/output_mask.png",
    "dataset": "MANGROVE_ROOTS",
    "output_format": ".ply"
}
```

#### **What It Does:**
1. **Reads the point cloud** (`pointcloud.txt`).
2. **Maps each point to a segmentation label** using `output_mask.png`.
3. **Attaches class labels (`class_id`) and colors (`RGB`)**.
4. **Saves the new point cloud** as a `.csv` or a `ply` file.

✔ **Output:**  
- `**_segmap.csv` or `**_segmap.ply` → Contains the original point cloud fields + segmentation class labels + r/g/b.

---

### 📌 Label Maps JSON Explanation (`label_maps.json`)

- `DATASETS`: Contains multiple datasets, each with its own colormap and class names.
- `COLOR_TO_INDEX`: Defines the **RGB-to-class-index mapping** for segmentation masks.
- `CLASS_NAMES`: Maps **class indices to human-readable names**.

#### **Example JSON Structure**
```json
{
    "DATASETS": {
        "MANGROVE_ROOTS": {
            "COLOR_TO_INDEX": { ... },
            "CLASS_NAMES": { ... }
        }
    }
}
```

#### **Adding a New Dataset**
1. Open `label_maps.json`.
2. Add a new key under `DATASETS` with the dataset name.
3. Define the **colormap (`COLOR_TO_INDEX`)** and **class names (`CLASS_NAMES`)**.
4. Example:
```json
{
    "DATASETS": {
        "MANGROVE_ROOTS": {
            "COLOR_TO_INDEX": { ... },
            "CLASS_NAMES": { ... }
        },
        "NEW_DATASET": { 
            "COLOR_TO_INDEX": {
                "0,0,0": 0,
                "255,0,0": 1,
                "0,255,0": 2,
                "0,0,255": 3,
                "255,255,0": 4
            },
            "CLASS_NAMES": {
                "0": "Background",
                "1": "Category 1",
                "2": "Category 2",
                "3": "Category 3",
                "4": "Category 4"
            }
        }
    }
}
```

---

## **🚀 Summary**
✔ **Convert** segmentation maps to grayscale masks.  
✔ **Visualize** class index masks with colors.  
✔ **Attach segmentation labels** to 3D point clouds.  
✔ **Easily configure datasets** in `label_maps.json`.  

