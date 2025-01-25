
# **Annotation Guideline for Forest and Mangrove Root Scene Semantic Segmentation**

This guideline is designed to ensure consistency and accuracy when labeling **Harvard Forest** and **Mangrove Root** scenes for semantic segmentation. It defines the labeling rules for specific categories in both environments to support ecological applications such as biomass estimation, structural analysis, and habitat modeling.

---

## **1. General Rules**
- **Dominance**: Label each point based on the dominant feature within its spatial neighborhood.
- **Precision**: Annotate boundaries as accurately as possible. Use zoom tools in annotation software for detailed labeling.
- **Overlapping Features**: Assign labels based on the visible feature, considering occlusions and structural prominence:
  - For forests: **Hierarchy**: Trunk > Large Branches > Leaves > Deadwood > Ground/Terrain > Bushes/Undergrowth > Miscellaneous.
  - For mangroves: **Hierarchy**: Roots > Trunk > Ground > Canopy > Water > Miscellaneous.
- **Occlusions**: For partially visible objects, label only the visible portions. For fully obscured areas, rely on the category of the covering object.

---

## **2. Label Definitions and Guidelines**

### **For Harvard Forest**

#### a. Ground/Terrain
- **Definition**: The forest floor, including soil, exposed terrain, and underlying vegetation-free surfaces.
- **Inclusions**:
  - Bare ground, muddy surfaces, or soil with minor debris.
  - Rocks or boulders as part of the terrain.
- **Exclusions**:
  - Grass, shrubs, or undergrowth covering the soil (label as **Bushes/Undergrowth**).
  - Fallen leaves or woody debris (label as **Deadwood** if significant).

#### b. Trunk
- **Definition**: Main vertical structures of trees used for tree metrics like DBH.
- **Inclusions**:
  - Tree trunks and large, exposed bases of trees.
- **Exclusions**:
  - Thin branches or secondary stems (label as **Large Branches**).
  - Detached logs (label as **Deadwood**).

#### c. Large Branches
- **Definition**: Primary branches of trees that are distinct from the trunk.
- **Inclusions**:
  - Exposed, visible large branches connected to the trunk.
- **Exclusions**:
  - Fine twigs or foliage-covered branches (label as **Leaves/Small Branches**).

#### d. Leaves/Small Branches (Canopy)
- **Definition**: Foliage and thin branches that form the canopy or understory vegetation.
- **Inclusions**:
  - Tree canopy leaves and small branches.
  - Shrub leaves forming part of the understory.
- **Exclusions**:
  - Fallen leaves on the ground (label as **Ground/Terrain**).

#### e. Deadwood/Logs
- **Definition**: Fallen or detached woody debris, including branches and logs.
- **Inclusions**:
  - Dead or decaying wood lying on the forest floor.
  - Logs used for ecological studies.
- **Exclusions**:
  - Dead branches attached to trees (label as **Trunk** or **Large Branches**).

#### f. Bushes/Undergrowth/Other Vegetation
- **Definition**: Shrubs, ferns, and grass forming the forest understory.
- **Inclusions**:
  - Dense undergrowth or ground-level vegetation.
- **Exclusions**:
  - Small trees or saplings (label appropriately based on height and structure).

#### g. Miscellaneous
- **Definition**: Non-natural or external objects in the scene.
- **Inclusions**:
  - People, LiDAR stands, or measurement tools.
  - Other artificial objects.


### **Harvard Forest Segmentation Table**

| Label Name                   | Segmentation Label | Color            |
|------------------------------|--------------------|------------------|
| Void                         | 0                  | Black ⚫ |
| Ground/Terrain               | 1                  | Purple 🟣 |
| Trunk                        | 2                  | Brown 🟤  |
| Large Branches               | 3                  | Orange 🟠 |
| Leaves/Small Branches (Canopy)| 4                 | Green 🟢 |
| Deadwood/Logs                | 5                  | Yellow 🟡 |
|Bushes/Undergrowth/Other Vegetation| 6             | Light blue 🔵 |
| Miscellaneous                | 7                  | white ⚪ |

---


### **For Mangrove Roots**

#### a. Ground
- **Definition**: Soil or muddy terrain beneath the mangroves.
- **Inclusions**:
  - Mudflats, sandy soil, or sediment layers.
- **Exclusions**:
  - Roots partially embedded in soil (label as **Roots**).

#### b. Roots
- **Definition**: All root structures of mangroves, categorized further into:
  - **Aerial Roots**: Above-ground roots like prop or stilt roots.
  - **Grounded Roots**: Roots partially embedded in soil.
  - **Species-Specific Roots**:
    - Pneumatophores of *Sonneratia alba*.
    - Prop roots of *Rhizophora* species.
    - Ribbon roots of *Xylocarpus granatum*.
- **Inclusions**:
  - All visible root structures.
- **Exclusions**:
  - Trunk-like structures (label as **Trunk**).

#### c. Trunk
- **Definition**: Vertical stems of mangroves supporting above-ground biomass.
- **Inclusions**:
  - Main upright structures of mangrove trees.
- **Exclusions**:
  - Low-lying aerial roots (label as **Roots**).

#### d. Canopy
- **Definition**: Foliage and small branches forming the upper structure of mangroves.
- **Inclusions**:
  - Leaves and small twigs visible above the trunk and roots.
- **Exclusions**:
  - Detached branches or leaves on the ground.

#### e. Water
- **Definition**: Reflective surfaces representing tidal or surrounding water bodies.
- **Inclusions**:
  - Open water beneath mangroves.
- **Exclusions**:
  - Submerged roots (label as **Roots**).

#### f. Miscellaneous
- **Definition**: Non-natural objects in the scene or unclear points.
- **Inclusions**:
  - Boats, researchers, or equipment visible in the scene.

### **Mangrove Roots Segmentation Table**

| Label Name                   | Segmentation Label | Color            |
|------------------------------|--------------------|------------------|
| Void                         | 0                  | Black ⚫ |
| Roots                        | 1                  | Yellow 🟡|
| Ground                       | 2                  | Purple 🟣|
| Trunk                        | 4                  | Brown 🟤 |
| Canopy                       | 5                  | Green 🟢 |
| Water                        | 6                  | Orange 🟠|
| Miscellaneous                | 3                  | White ⚪ |
---


## **3. Dataset Checklist**
Before finalizing annotation:
- Verify all categories are consistent with the guidelines.
- Confirm clear boundaries between adjacent categories.
- Validate segmentation maps for errors (e.g., mislabeled points or missing data).
- Ensure both forest and mangrove-specific categories are represented.

---

Extra Note: [emoji-keys](https://1000logos.net/emoji-copy-and-paste/) # (Emoji Key: 🔵🟢 🟤 🟣 🟠 🟡 ⚫ ⚪)


# Color Swatches

Here are the color swatches displayed in a Markdown file:

<div style="display: flex; flex-wrap: wrap; gap: 10px;">
    <div style="width: 100px; height: 100px; border-radius: 50%; background-color: #ADD8E6; display: flex; align-items: center; justify-content: center; font-size: 24px; color: white; text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);">🔵</div> <!-- Light Blue -->
    <div style="width: 100px; height: 100px; border-radius: 50%; background-color: #00FF00; display: flex; align-items: center; justify-content: center; font-size: 24px; color: white; text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);">🟢</div> <!-- Green -->
    <div style="width: 100px; height: 100px; border-radius: 50%; background-color: #A52A2A; display: flex; align-items: center; justify-content: center; font-size: 24px; color: white; text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);">🟤</div> <!-- Brown -->
    <div style="width: 100px; height: 100px; border-radius: 50%; background-color: #800080; display: flex; align-items: center; justify-content: center; font-size: 24px; color: white; text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);">🟣</div> <!-- Purple -->
    <div style="width: 100px; height: 100px; border-radius: 50%; background-color: #FFA500; display: flex; align-items: center; justify-content: center; font-size: 24px; color: white; text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);">🟠</div> <!-- Orange -->
    <div style="width: 100px; height: 100px; border-radius: 50%; background-color: #FFFF00; display: flex; align-items: center; justify-content: center; font-size: 24px; color: black; text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);">🟡</div> <!-- Yellow -->
    <div style="width: 100px; height: 100px; border-radius: 50%; background-color: #000000; display: flex; align-items: center; justify-content: center; font-size: 24px; color: white; text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);">⚫</div> <!-- Black -->
    <div style="width: 100px; height: 100px; border-radius: 50%; background-color: #FFFFFF; display: flex; align-items: center; justify-content: center; font-size: 24px; color: black; text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);">⚪</div> <!-- White -->
</div>
