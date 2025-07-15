# CBL GBL Processing Guide

Convert Compact Biomass Lidar (CBL) `.gbl` files to text.
<!-- and SPD/LAS formats. -->

## Usage

```bash
python CBL_GBL_Processing_UMB.py --inFolder <path_to_gbl_files> [options]
```

<!-- ## Required Arguments

- `--inFolder`: Path to folder containing `.gbl` files -->

<!-- ## Optional Arguments

- `--cblVersion`: CBL instrument version (1 or 2, default: 2)
- `--agh`: Above ground height of sensor in meters (default: 1.2)
- `--verbose`, `-v`: Enable verbose output -->

<!-- ## Examples

### Basic usage
```bash
python CBL_GBL_Processing_UMB.py --inFolder /path/to/gbl/files_folder
``` -->

<!-- ### CBL1 instrument with custom height
```bash
python CBL_GBL_Processing_UMB.py --inFolder ./data/cbl1_scans --cblVersion 1 --agh 1.5
``` -->

### Verbose processing
```bash
python CBL_GBL_Processing_UMB.py --inFolder /path/to/gbl/files_folder --verbose
```

## Output Files

For each `.gbl` input file, the script generates:
- `.txt`: Text format with point coordinates and attributes
<!-- - `.spd`: SPD format (if spdlib available)
- `.las`: LAS format (if spdlib available) -->

## Dependencies

- Required: `numpy`, `pathlib`
<!-- - Optional: `spdpy` (for SPD/LAS output) -->
