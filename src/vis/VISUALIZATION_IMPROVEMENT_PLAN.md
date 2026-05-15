# Visualization Scripts Improvement Plan

## Overview
Improve `vis_bbox_img.py` and `vis_bbox_pts.py` to create unified visualization tools that can:
- Accept inputs from the same data sources
- Visualize both ground truth and prediction bounding boxes
- Support multiple prediction files for comparison
- Project bounding boxes on both camera images and point clouds

## Current State Analysis

### `vis_bbox_img.py`
- ✅ Loads GT from pkl file
- ✅ Projects 3D bboxes to 2D image
- ❌ Hardcoded file paths
- ❌ No prediction support
- ❌ No command-line arguments
- ❌ Duplicate bbox creation code

### `vis_bbox_pts.py`
- ✅ Loads GT from pkl file
- ✅ Converts camera to LiDAR coordinates
- ✅ Visualizes on point cloud
- ✅ Has hardcoded prediction example
- ❌ Hardcoded file paths
- ❌ No command-line arguments
- ❌ Prediction format not flexible

## Proposed Improvements

### 1. Common Utilities Module (`vis_utils.py`)
Create a shared utility module with:
- **Coordinate conversion functions**:
  - `cam_to_lidar_coords(x, y, z)` - Convert camera to LiDAR coordinates
  - `cam_to_lidar_yaw(yaw_cam)` - Convert camera yaw to LiDAR yaw
  - `camera_boxes_to_lidar(camera_boxes)` - Convert CameraInstance3DBoxes to LiDARInstance3DBoxes
- **Data loading functions**:
  - `load_gt_from_pkl(pkl_path, file_idx)` - Load ground truth from pkl file
  - `load_prediction_from_json(json_path)` - Load prediction from JSON file
  - `parse_prediction_dict(pred_dict)` - Parse prediction dict to bbox format
- **Color schemes**:
  - GT color: Green (0, 255, 0)
  - Prediction colors: List of distinct colors (Red, Blue, Yellow, Cyan, Magenta, etc.)
  - Color assignment function for multiple predictions

### 2. Enhanced `vis_bbox_img.py`
**Features to add:**
- Command-line argument parsing (argparse):
  - `--file_idx`: File index (e.g., '000000')
  - `--gt_pkl`: Path to ground truth pkl file
  - `--image`: Path to image file
  - `--pred`: Path(s) to prediction JSON file(s) (can specify multiple)
  - `--pred_labels`: Optional labels for predictions (for legend/identification)
  - `--output`: Optional output path to save visualization (instead of showing)
  - `--camera`: Camera ID (default: 'CAM2')
- Load ground truth from pkl
- Load one or more prediction files
- Visualize GT bboxes in green
- Visualize each prediction in different colors
- Support saving output image
- Clean up duplicate code

**Data flow:**
1. Load GT from pkl → CameraInstance3DBoxes
2. Load predictions from JSON → Convert to CameraInstance3DBoxes (if needed)
3. Project all bboxes to 2D image
4. Draw with different colors

### 3. Enhanced `vis_bbox_pts.py`
**Features to add:**
- Command-line argument parsing (same as above, but with `--lidar` instead of `--image`):
  - `--file_idx`: File index
  - `--gt_pkl`: Path to ground truth pkl file
  - `--lidar`: Path to point cloud (.bin) file
  - `--pred`: Path(s) to prediction JSON file(s)
  - `--pred_labels`: Optional labels for predictions
  - `--output`: Optional output path to save visualization
- Load ground truth from pkl → Convert to LiDAR coordinates
- Load predictions from JSON → Ensure LiDAR format
- Visualize GT bboxes in green
- Visualize each prediction in different colors
- Support saving output
- Remove hardcoded prediction example

**Data flow:**
1. Load GT from pkl → CameraInstance3DBoxes → Convert to LiDARInstance3DBoxes
2. Load predictions from JSON → LiDARInstance3DBoxes (if not already)
3. Load point cloud
4. Draw all bboxes on point cloud with different colors

### 4. Prediction File Format
**Expected JSON format:**
```json
{
  "labels_3d": [0, 0, 1, 2],
  "scores_3d": [0.88, 0.23, 0.57, 0.45],
  "bboxes_3d": [
    [x, y, z, l, w, h, yaw],
    ...
  ],
  "box_type_3d": "LiDAR"  // or "Camera"
}
```

**Handling:**
- If `box_type_3d` is "Camera", convert to LiDAR for point cloud visualization
- If `box_type_3d` is "LiDAR", use directly for point cloud, convert for image visualization
- Support both formats flexibly

### 5. Color Scheme
- **Ground Truth**: Green `(0, 255, 0)`
- **Prediction 1**: Red `(255, 0, 0)`
- **Prediction 2**: Blue `(0, 0, 255)`
- **Prediction 3**: Yellow `(255, 255, 0)`
- **Prediction 4**: Cyan `(0, 255, 255)`
- **Prediction 5**: Magenta `(255, 0, 255)`
- **Prediction 6+**: Cycle through colors or use distinct palette

### 6. Implementation Steps

1. **Create `vis_utils.py`** with shared utilities
2. **Refactor `vis_bbox_img.py`**:
   - Add argparse
   - Use utilities from `vis_utils.py`
   - Support multiple predictions
   - Add output saving option
3. **Refactor `vis_bbox_pts.py`**:
   - Add argparse
   - Use utilities from `vis_utils.py`
   - Support multiple predictions
   - Add output saving option
   - Remove hardcoded examples
4. **Test with sample data**:
   - Single prediction
   - Multiple predictions
   - Both image and point cloud views

### 7. Usage Examples

**Image visualization:**
```bash
python vis_bbox_img.py \
  --file_idx 000000 \
  --gt_pkl /path/to/kitti_infos_train_000000.pkl \
  --image /path/to/000000.png \
  --pred /path/to/pred1.json /path/to/pred2.json \
  --pred_labels "Baseline" "Attack" \
  --output output.png
```

**Point cloud visualization:**
```bash
python vis_bbox_pts.py \
  --file_idx 000000 \
  --gt_pkl /path/to/kitti_infos_train_000000.pkl \
  --lidar /path/to/000000.bin \
  --pred /path/to/pred1.json /path/to/pred2.json \
  --pred_labels "Baseline" "Attack" \
  --output output.png
```

## Benefits
1. **Unified interface**: Both scripts use same argument structure
2. **Flexible**: Support multiple predictions for comparison
3. **Reusable**: Common utilities reduce code duplication
4. **Configurable**: No hardcoded paths
5. **Extensible**: Easy to add new features or formats

