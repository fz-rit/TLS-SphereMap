
# Annotation Guideline for Forest Scene Semantic Segmentation

This guideline is designed to ensure consistency and accuracy when labeling forest scenes for semantic segmentation. It defines the labeling rules for five categories: **Leaves**, **Bark (trunk and branch)**, **Grass and Shrubs**, **Soil**, and **Miscellaneous**.

---

## 1. General Rules
- **Dominance**: Label each pixel based on the dominant feature visible in that pixel.
- **Precision**: Annotate boundaries as accurately as possible. Use zoom-in tools in annotation software to ensure fine detail.
- **Overlapping Features**: Prioritize the more prominent feature in the hierarchy:
  - **Hierarchy**: Bark > Leaves > Grass/Shrubs > Soil > Miscellaneous.
- **Occlusions**: Label visible portions of partially occluded objects. If fully obscured, label based on the covering object.

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

### b. Bark (Trunk and Branch)
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

### c. Grass and Shrubs
- **Definition**: Pixels corresponding to ground-level green vegetation, including grasses, shrubs, and any understory plants.
- **Inclusions**:
  - Grass patches, even if sparse.
  - Low shrubs and small bushes.
  - Ground-cover vegetation like moss or vines.
- **Exclusions**:
  - Trees or tree canopy (label as "Leaves" or "Bark").
  - Fallen branches or deadwood (label as "Miscellaneous").
- **Edge Cases**:
  - For areas where grass and shrubs overlap with soil, prioritize "Grass and Shrubs" if vegetation is dominant.

---

### d. Soil (Including Rocks)
- **Definition**: Pixels corresponding to exposed ground, including soil, rocks, and areas without vegetation.
- **Inclusions**:
  - Bare ground with or without vegetation debris.
  - Rocks and boulders.
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
- **Exclusions**:
  - Rocks or natural terrain (label as "Soil").
  - Fallen branches or logs (label as "Miscellaneous" or exclude depending on context).
- **Edge Cases**:
  - If objects are partially visible, label only the visible parts.

---

## 3. Practical Annotation Tips
- **Labeling Software**: Use tools like LabelImg, Labelbox, or CVAT for polygon-based or pixel-level annotations.
- **Zooming**: Zoom in for detailed boundary labeling, especially around small objects like branches.
- **Save Progress**: Frequently save your annotations to avoid losing work.

---

## 4. Mask Encoding
When exporting segmentation masks, assign unique pixel values to each label:

| Label              | Pixel Value in Mask |
|--------------------|---------------------|
| Leaves             | 1                   |
| Bark (trunk/branch)| 2                   |
| Grass and Shrubs   | 3                   |
| Soil (including rocks) | 4              |
| Miscellaneous      | 5                   |

---

## 5. Dataset Checklist
Before completing annotation:
- Review for missing or ambiguous labels.
- Verify boundary consistency between adjacent categories.
- Confirm each label is adequately represented in the dataset.
