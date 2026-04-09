# H&E ↔ DAPI Cell Alignment

Align each H&E cell to candidate DAPI/CosMx cells using shared pixel coordinates (`x_center_pixel`, `y_center_pixel`) and multi-direction patch matching.

## What changed for your data format

This script now directly supports CSVs like:

```csv
,x_center_pixel,y_center_pixel,center_radius_pixel
0,4187.0,387.5,4.0
1,4396.0,388.5,6.0
```

- Uses `x_center_pixel` / `y_center_pixel` by default.
- If `cell_id` is missing, it uses the unnamed first column (`0,1,2,...`) as ID.
- Can optionally use `center_radius_pixel` to set a dynamic patch radius (`--use-cell-radius`).

## Matching workflow

For each H&E cell:
1. Query nearby DAPI candidates within `--search-radius`.
2. For each candidate pair, crop 5 directional patches in both images from corresponding coordinates:
   - center, up, down, left, right
3. Direction offset is `--move-radius` pixels.
4. Compare each patch pair using a selectable 2D similarity metric:
   - `ncc` (default): normalized cross-correlation
   - `ssim`: structural similarity (single-window)
   - `cosine`: cosine over normalized patch vectors
5. Aggregate directional scores:

`final_score = mean(direction_scores) - consistency_penalty * std(direction_scores)`

6. Return top-k DAPI candidates per H&E cell.

## Why NCC as default (vs cosine)

- **Cosine** on flattened patches is valid and often strong.
- **NCC** is effectively the 2D intensity-pattern correlation after normalization and is very robust to brightness/contrast shifts.
- **SSIM** can help when structure is more important than raw intensity, but may be less stable across modality differences.

For H&E vs DAPI cross-modality matching, start with `ncc`, then compare with `ssim` on a validation subset.

## Run

```bash
python align_cells.py \
  --he-image he_image.tif \
  --dapi-image dapi_image.tif \
  --he-cells dapipose_cellpose_segment_H\&E_label_location_save.csv \
  --dapi-cells dapipose_cellpose_segment_DAPI_label_location_save.csv \
  --search-radius 30 \
  --move-radius 4 \
  --top-k 3 \
  --metric ncc \
  --use-cell-radius \
  --radius-scale 2.0 \
  --output alignment_results.json
```

If you do not use per-cell radius, provide fixed patch radius:

```bash
python align_cells.py ... --patch-radius 12
```

## Output

Each JSON item includes:
- `he_cell_id`
- `dapi_cell_id`
- `aggregated_score`
- `metric`
- `patch_radius_used`
- `direction_scores`
- `he_xy`
- `dapi_xy`
