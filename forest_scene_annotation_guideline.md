
# Annotation Guideline for Forest Scene Semantic Segmentation

This guideline is designed to ensure consistency and accuracy when labeling forest scenes for semantic segmentation. It defines the labeling rules for five categories: **Leaves**, **Bark**, **Soil**, and **Miscellaneous**.

---

## 1. General Rules
- **Dominance**: Label each pixel based on the dominant feature visible in that pixel.
- **Precision**: Annotate boundaries as accurately as possible. Use zoom-in tools in annotation software to ensure fine detail.
- **Overlapping Features**: Prioritize the more prominent feature in the hierarchy:
  - **Hierarchy**: Bark > Leaves > Miscellaneous > Soil.
- **Occlusions**: Label visible portions of partially occluded objects. If fully obscured, label based on the covering object.
### **Practical Tips**  
- **Priority Hierarchy**: `Miscellaneous > Bark > Leaves > Soil` or `Miscellaneous > Leaves > Bark > Soil`.  
- **Label Order**: Reverse the priority hierarchy: Soil > Leaves or Bark (whichever is dominant) > Miscellaneous. Be more lenient with earlier categories (tolerate more false positives).  
- **Colorization**: Follow the priority hierarchy: Miscellaneous > Leaves or Bark > Soil.  

---

## 2. Label Definitions and Guidelines

### a. Leaves
- **Definition**: Pixels corresponding to tree or shrub foliage, including both broadleaf and needle-like structures.
- **Inclusions**:
  - Tree canopy.
  - Shrubs and understory leaves.
  - Dead leaves still attached to plants.
- **Exclusions**:
  - Fallen leaves on the ground (label as "Soil").
  - Overlapping branches (label as "Bark").
- **Edge Cases**:
  - If mixed with thin twigs, label as "Leaves" unless twigs are prominent.

---

### b. Bark
- **Definition**: Pixels corresponding to woody parts of the tree, including trunks and branches.
- **Inclusions**:
  - Main trunk and exposed branches.
  - Visible portions of bark-covered woody structures.
- **Exclusions**:
  - Thin twigs heavily mixed with leaves (label as "Leaves").
  - Deadwood lying on the ground (label as "Miscellaneous" or exclude depending on context).
- **Edge Cases**:
  - For branches partially obscured by leaves, label the visible parts as "Bark."
  - Include both live and dead branches still attached to trees.

---

### c. Soil
- **Definition**: Pixels corresponding to exposed ground, including soil, rocks, and areas without vegetation.
- **Inclusions**:
  - Bare ground with or without vegetation debris.
  - Dry or muddy soil.
- **Exclusions**:
  - Grass or shrubs covering soil (label as "Grass and Shrubs").
  - Fallen leaves (consider labeling with "Soil" if mixed into the ground cover).
- **Edge Cases**:
  - For rocky terrain, include both rocks and soil under "Soil."

---

### e. Miscellaneous
- **Definition**: Pixels corresponding to non-natural objects or items not covered by other labels.
- **Inclusions**:
  - Human-made objects (e.g., targets, LiDAR stands, measurement tools).
  - People or animals in the scene.
  - Rocks and boulders.
- **Exclusions**:
  - Fallen branches or logs (label as "Miscellaneous" or exclude depending on context).
- **Edge Cases**:
  - If objects are partially visible, label only the visible parts.

---

## 3. Mask Encoding
The segmentation map is provided in two formats:  
1. A **colorized RGB PNG file**, where each label is represented by a unique color.  
2. A **monochrome TIFF file**, where grayscale values correspond to specific labels derived from the RGB image.  

| Label Name        | Segmentation Label | RGB Color           | Color Bar                                                                                   | Grayscale Value |
|--------------------|---------------------|----------------------|---------------------------------------------------------------------------------------------|------------------|
| Leaves            | 1                   | [0, 166, 81]        | ![#00a651](https://via.placeholder.com/15/00a651/000000?text=+)                              | 122              |
| Bark              | 2                   | [0, 174, 239]       | ![#00aef0](https://via.placeholder.com/15/00aef0/000000?text=+)                              | 141              |
| Soil              | 3                   | [255, 242, 0]       | ![#fff200](https://via.placeholder.com/15/fff200/000000?text=+)                              | 233              |
| Miscellaneous     | 4                   | [237, 28, 36]       | ![#ed1c24](https://via.placeholder.com/15/ed1c24/000000?text=+)                              | 100              |


---

## 4. Dataset Checklist
Before completing annotation:
- Review for missing or ambiguous labels.
- Verify boundary consistency between adjacent categories.
- Confirm each label is adequately represented in the dataset.
