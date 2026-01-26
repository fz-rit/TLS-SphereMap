# How to Use replot_from_saved.py

This script allows you to quickly regenerate plots from previously computed results without rerunning expensive density calculations.

## Workflow

### 1. Run Initial Analysis
First, run your density analysis which will automatically save intermediate results:

```bash
python pcd_profiler/density_vs_range_height.py
```

This will create result files in your configured output directory:
- `results.npz` - Density vs range numerical data
- `results.json` - Metadata
- `results_vertical.npz` - Vertical density numerical data  
- `results_vertical.json` - Vertical metadata

### 2. Replot with Different Settings

Once you have saved results, you can regenerate plots with different parameters:

#### Basic usage (default settings):
```bash
python pcd_profiler/replot_from_saved.py /home/fzhcis/Downloads/temp/results.npz
```

This creates a new directory `_replot` with regenerated plots.

#### Custom output directory:
```bash
python pcd_profiler/replot_from_saved.py /home/fzhcis/Downloads/temp/results.npz --output-dir /home/fzhcis/Downloads/temp/replot_custom
```

#### Custom figure size and DPI:
```bash
# Larger figure, lower DPI
python pcd_profiler/replot_from_saved.py /home/fzhcis/Downloads/temp/results.npz --figsize 16 10 --dpi 150

# Publication quality
python pcd_profiler/replot_from_saved.py /home/fzhcis/Downloads/temp/results.npz --figsize 8 6 --dpi 600
```

#### Plot only specific types:
```bash
# Only density vs range plot
python pcd_profiler/replot_from_saved.py /home/fzhcis/Downloads/temp/results.npz --plot-type density

# Only vertical density violin plot
python pcd_profiler/replot_from_saved.py /home/fzhcis/Downloads/temp/results.npz --plot-type vertical

# Both plots (default)
python pcd_profiler/replot_from_saved.py /home/fzhcis/Downloads/temp/results.npz --plot-type both
```

#### Combined example:
```bash
python pcd_profiler/replot_from_saved.py \
    /home/user/Downloads/temp/results.npz \
    --output-dir /home/user/Downloads/temp/replot_highres \
    --figsize 14 8 \
    --dpi 600 \
    --plot-type both
```

## All Available Options

- `results_path` (required): Path to saved results (.npz or .json file)
- `--output-dir`: Directory to save new plots (default: creates `_replot` directory)
- `--figsize`: Figure size as two numbers: width height (default: 12 6)
- `--dpi`: DPI for saved figures (default: 300)
- `--plot-type`: Which plots to generate: `density`, `vertical`, or `both` (default: both)

## Benefits

✅ **Fast**: No need to recompute expensive density calculations  
✅ **Flexible**: Easily try different figure sizes and DPI settings  
✅ **Iterative**: Perfect for preparing publication-quality figures  
✅ **Efficient**: Generate multiple versions without reprocessing data

## Example Workflow

```bash
# 1. Run initial analysis (takes time)
python pcd_profiler/density_vs_range_height.py

# 2. Check results in output directory
# Results saved to: /home/user/output/results.npz

# 3. Try different visualizations (fast)
python pcd_profiler/replot_from_saved.py /home/user/output/results.npz \
    --output-dir /home/user/output/version1 --figsize 10 6 --dpi 300

python pcd_profiler/replot_from_saved.py /home/user/output/results.npz \
    --output-dir /home/user/output/version2 --figsize 16 10 --dpi 150

python pcd_profiler/replot_from_saved.py /home/user/output/results.npz \
    --output-dir /home/user/output/publication --figsize 8 6 --dpi 600
```

## Tips

- Keep your `.npz` and `.json` files together in the same directory
- For vertical density plots, both `results.npz` and `results_vertical.npz` need to exist
- Use lower DPI (150-200) for quick previews, higher DPI (600) for publications
- Common figure sizes:
  - `12 6` - Default, good for presentations
  - `8 6` - Compact, good for papers with limited space
  - `14 10` - Large, good for posters
  - `16 9` - Widescreen format
