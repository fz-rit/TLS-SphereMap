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
python convert_color_to_mask.py

# [Optional] Step 2: Visualize the generated class index mask
python visualize_mask.py -i ../examples/seg_map_harvard_33_02_mask.png

# Step 3: Attach segmentation labels to a 3D point cloud
python attach_segmap_to_points.py
```
✔ **Generated Files:**
- `seg_map_harvard_33_02_mask.png` → Grayscale class index mask.
- `visualized_mask.png` → Colorized visualization of the mask.
- `point_cloud_with_segmentation.csv` → Point cloud with attached segmentation labels.

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

## **📔 Summary**
✔ **Convert** segmentation maps to grayscale masks.  
✔ **Visualize** class index masks with colors.  
✔ **Attach segmentation labels** to 3D point clouds.  
✔ **Easily configure datasets** in `label_maps.json`.  

