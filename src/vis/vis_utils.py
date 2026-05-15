"""
Shared utilities for visualization scripts.
Contains coordinate conversion, data loading, and color management functions.
"""
import torch
import numpy as np
from mmdet3d.structures import LiDARInstance3DBoxes, CameraInstance3DBoxes
from mmengine import load


# ============================================================================
# Coordinate Conversion Functions
# ============================================================================

def cam_to_lidar_coords(x, y, z):
    """Convert camera coordinates to LiDAR coordinates.
    
    Args:
        x, y, z: Camera coordinates
        
    Returns:
        tuple: (x_lidar, y_lidar, z_lidar)
    """
    return z, -x, -y


def cam_to_lidar_yaw(yaw_cam):
    """Convert camera yaw angle to LiDAR yaw angle.
    
    Args:
        yaw_cam: Yaw angle in camera frame
        
    Returns:
        float: Yaw angle in LiDAR frame
    """
    return -yaw_cam


def camera_boxes_to_lidar(camera_boxes: CameraInstance3DBoxes):
    """Convert CameraInstance3DBoxes to LiDARInstance3DBoxes.
    
    Args:
        camera_boxes: Bounding boxes in camera coordinates
        
    Returns:
        LiDARInstance3DBoxes: Bounding boxes in LiDAR coordinates
    """
    boxes = camera_boxes.tensor.cpu().numpy()
    centers = boxes[:, :3]
    dims = boxes[:, 3:6]
    yaws = boxes[:, 6]

    # Convert coords and yaw
    centers_lidar = np.stack([*cam_to_lidar_coords(centers[:, 0], centers[:, 1], centers[:, 2])], axis=-1)
    yaws_lidar = cam_to_lidar_yaw(yaws)

    # Reorder dims from camera (w, h, l) → LiDAR (l, w, h)
    dims_reordered = dims[:, [2, 0, 1]]

    boxes_lidar = np.concatenate([centers_lidar, dims_reordered, yaws_lidar[:, None]], axis=1)
    return LiDARInstance3DBoxes(torch.tensor(boxes_lidar, dtype=torch.float32))


def lidar_boxes_to_camera(lidar_boxes: LiDARInstance3DBoxes):
    """Convert LiDARInstance3DBoxes to CameraInstance3DBoxes.
    
    Args:
        lidar_boxes: Bounding boxes in LiDAR coordinates
        
    Returns:
        CameraInstance3DBoxes: Bounding boxes in camera coordinates
    """
    boxes = lidar_boxes.tensor.cpu().numpy()
    centers = boxes[:, :3]
    dims = boxes[:, 3:6]
    yaws = boxes[:, 6]

    # Convert coords: LiDAR (x, y, z) → Camera (x, y, z)
    # Inverse of cam_to_lidar_coords: (z, -x, -y) → (x, y, z)
    centers_cam = np.stack([-centers[:, 1], -centers[:, 2], centers[:, 0]], axis=-1)
    yaws_cam = -yaws  # Inverse of cam_to_lidar_yaw

    # Reorder dims from LiDAR (l, w, h) → Camera (w, h, l)
    dims_reordered = dims[:, [1, 2, 0]]

    boxes_cam = np.concatenate([centers_cam, dims_reordered, yaws_cam[:, None]], axis=1)
    return CameraInstance3DBoxes(torch.tensor(boxes_cam, dtype=torch.float32))


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_gt_from_pkl(pkl_path, file_idx=None):
    """Load ground truth bounding boxes from pkl file.
    
    Args:
        pkl_path: Path to pkl file
        file_idx: Optional file index to extract specific sample
        
    Returns:
        dict: Contains 'bboxes_3d' (CameraInstance3DBoxes), 'cam2img', and 'data_list'
    """
    info_file = load(pkl_path)
    
    # If file_idx is provided, try to find matching data
    if file_idx is not None:
        # Try to find data matching file_idx
        for data_item in info_file.get('data_list', []):
            # Check if this matches the file_idx (implementation depends on data structure)
            # For now, just take first item if file_idx is provided
            if len(info_file.get('data_list', [])) > 0:
                data_item = info_file['data_list'][0]
                break
    else:
        data_item = info_file['data_list'][0] if info_file.get('data_list') else None
    
    if data_item is None:
        raise ValueError("No data found in pkl file")
    
    # Extract cam2img matrix
    cam2img = np.array(data_item['images']['CAM2']['cam2img'], dtype=np.float32)
    
    # Extract bounding boxes
    bboxes_3d = []
    for instance in data_item.get('instances', []):
        if instance.get('bbox_label', 0) == -1:
            continue
        bboxes_3d.append(instance['bbox_3d'])
    
    if len(bboxes_3d) == 0:
        gt_bboxes_3d = CameraInstance3DBoxes(torch.zeros((0, 7), dtype=torch.float32))
    else:
        gt_bboxes_3d = CameraInstance3DBoxes(np.array(bboxes_3d, dtype=np.float32))
    
    return {
        'bboxes_3d': gt_bboxes_3d,
        'cam2img': cam2img,
        'data_list': info_file.get('data_list', [])
    }


def load_prediction_from_json(json_path, score_threshold=None):
    """Load prediction from JSON file.
    
    Expected format:
    {
        "labels_3d": [0, 0, 1, 2],
        "scores_3d": [0.88, 0.23, 0.57, 0.45],
        "bboxes_3d": [[x, y, z, l, w, h, yaw], ...],
        "box_type_3d": "LiDAR"  # or "Camera"
    }
    
    Args:
        json_path: Path to JSON file
        score_threshold: Optional minimum score threshold to filter predictions
        
    Returns:
        dict: Contains 'bboxes_3d' (LiDARInstance3DBoxes or CameraInstance3DBoxes),
              'labels_3d', 'scores_3d', 'box_type_3d'
    """
    import json
    
    with open(json_path, 'r') as f:
        pred_dict = json.load(f)
    
    bboxes_3d = np.array(pred_dict['bboxes_3d'], dtype=np.float32)
    box_type = pred_dict.get('box_type_3d', 'LiDAR')
    
    if box_type == 'LiDAR':
        bboxes_3d = LiDARInstance3DBoxes(torch.tensor(bboxes_3d, dtype=torch.float32))
    elif box_type == 'Camera':
        bboxes_3d = CameraInstance3DBoxes(torch.tensor(bboxes_3d, dtype=torch.float32))
    else:
        raise ValueError(f"Unknown box_type_3d: {box_type}")
    
    pred_data = {
        'bboxes_3d': bboxes_3d,
        'labels_3d': pred_dict.get('labels_3d', []),
        'scores_3d': pred_dict.get('scores_3d', []),
        'box_type_3d': box_type
    }
    
    # Apply score threshold filtering if specified
    if score_threshold is not None:
        pred_data = filter_predictions_by_score(pred_data, score_threshold)
    
    return pred_data


# ============================================================================
# Color Management
# ============================================================================

# Color scheme: GT in blue, predictions in different colors
GT_COLOR = (0, 0, 255)  # Blue
GT_COLOR_INVERTED = (255, 255, 0) # Yellow (inverse of Blue, for lidar visualization that will be inverted)
PREDICTION_COLORS = [
    (255, 0, 0),      # Red
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (0, 255, 255),    # Cyan
    (255, 0, 255),    # Magenta
    (255, 165, 0),    # Orange
    (128, 0, 128),    # Purple
    (255, 192, 203),  # Pink
]

# KITTI class label to color mapping
# Based on common KITTI dataset labels: 0=Pedestrian, 1=Cyclist, 2=Car, 3=Van, 4=Truck, etc.
# Note: Green is reserved for GT bboxes, so it's not used here
CLASS_LABEL_COLORS = {
    0: (0, 0, 255),      # Blue - Pedestrian
    1: (255, 255, 0),    # Yellow - Cyclist (changed from green)
    2: (255, 0, 0),      # Red - Car
    3: (255, 165, 0),    # Orange - Van
    4: (255, 192, 203),  # Pink - Truck
    5: (128, 0, 128),    # Purple - Person_sitting
    6: (0, 255, 255),    # Cyan - Tram
    7: (200, 200, 200),  # Light Gray - Misc (changed from yellow since yellow is now used for Cyclist)
}


def get_class_color(label):
    """Get color for a class label.
    
    Args:
        label: Class label (integer)
        
    Returns:
        tuple: RGB color tuple
    """
    return CLASS_LABEL_COLORS.get(label, (255, 255, 255))  # Default to white if unknown


def get_colors_from_labels(labels):
    """Get colors for bounding boxes based on their class labels.
    
    Args:
        labels: List of class labels
        
    Returns:
        list: List of RGB color tuples, one per label
    """
    return [get_class_color(label) for label in labels]


def get_prediction_color(pred_idx):
    """Get color for a prediction file by index (for distinguishing multiple prediction files).
    
    Args:
        pred_idx: Index of prediction file (0-based)
        
    Returns:
        tuple: RGB color tuple
    """
    return PREDICTION_COLORS[pred_idx % len(PREDICTION_COLORS)]


def get_red_gradient_color(pred_idx, total_preds, orange=(255, 165, 0), red=(255, 0, 0)):
    """Get a red gradient color based on prediction file index.
    
    Creates a gradient from orange (first file) to red (last file).
    
    Args:
        pred_idx: Index of prediction file (0-based)
        total_preds: Total number of prediction files
        orange: RGB tuple for orange color (default: (255, 165, 0))
        red: RGB tuple for red color (default: (255, 0, 0))
        
    Returns:
        tuple: RGB color tuple for this prediction file
    """
    if total_preds <= 1:
        return red
    
    # Calculate interpolation factor (0.0 for first, 1.0 for last)
    t = pred_idx / (total_preds - 1)
    
    # Interpolate between orange and red
    r = int(orange[0] + t * (red[0] - orange[0]))
    g = int(orange[1] + t * (red[1] - orange[1]))
    b = int(orange[2] + t * (red[2] - orange[2]))
    
    return (r, g, b)


def get_delay_color(delay, inverted=False):
    """Get a hardcoded color for a specific delay value.
    
    Color scheme: Delay 0 (greenish) -> Delay 5 (red)
    - Delay 0: Yellow-green/Lime (greenish but not green, since green is for GT)
    - Delay 1: Yellow
    - Delay 2: Orange
    - Delay 3: Orange-red
    - Delay 4: Red-orange
    - Delay 5: Red (highest misalignment)
    
    Args:
        delay: Delay value (0-5)
        inverted: If True, returns inverted colors (for lidar visualization that will be inverted)
                  Inverted colors will match camera colors after inversion
        
    Returns:
        tuple: RGB color tuple for this delay
    """
    # Hardcoded colors for delays 0-5
    # Colors progress from greenish (delay 0) to red (delay 5)
    delay_colors = {
        0: (173, 255, 47),   # Yellow-green / Lime (greenish but not green)
        1: (255, 255, 0),    # Yellow
        2: (255, 165, 0),    # Orange
        3: (255, 140, 0),    # Dark orange
        4: (255, 69, 0),     # Red-orange
        5: (255, 0, 0),      # Red (highest misalignment)
    }
    
    # Inverted colors (for lidar visualization that will be inverted)
    # These will match camera colors after inversion
    inverted_delay_colors = {
        0: (82, 0, 208),     # Purple (inverse of Lime)
        1: (0, 0, 255),      # Blue (inverse of Yellow)
        2: (0, 90, 255),    # Cyan-blue (inverse of Orange)
        3: (0, 115, 255),    # Cyan (inverse of Dark orange)
        4: (0, 186, 255),    # Light cyan (inverse of Red-orange)
        5: (0, 255, 255),    # Cyan (inverse of Red)
    }
    
    # Clamp delay to valid range
    delay = max(0, min(5, int(delay)))
    
    if inverted:
        return inverted_delay_colors[delay]
    else:
        return delay_colors[delay]


def extract_delay_from_filename(filename, delay_mode='lidar'):
    """Extract delay value from prediction filename.
    
    Filename format: {delay_image}_{delay_lidar}_{image_file:06d}_{lidar_file:06d}.json
    
    Args:
        filename: Prediction filename or full path
        delay_mode: Delay mode - 'lidar', 'image', or 'both'
        
    Returns:
        int: Delay value (0-5)
    """
    import os
    # Extract just the filename if full path is provided
    basename = os.path.basename(filename)
    # Remove extension
    basename = basename.replace('.json', '')
    
    # Split by underscore to get delay values
    parts = basename.split('_')
    if len(parts) >= 2:
        try:
            delay_image = int(parts[0])
            delay_lidar = int(parts[1])
            
            if delay_mode == 'lidar':
                return delay_lidar
            elif delay_mode == 'image':
                return delay_image
            elif delay_mode == 'both':
                # For 'both' mode, both delays should be the same, use either
                return max(delay_image, delay_lidar)
            else:
                return max(delay_image, delay_lidar)  # Default to max
        except ValueError:
            return 0  # Default to 0 if parsing fails
    
    return 0  # Default to 0 if format is unexpected


def get_bbox_colors(num_bboxes, color):
    """Get list of colors for multiple bboxes.
    
    Args:
        num_bboxes: Number of bounding boxes
        color: Single color tuple or list of colors
        
    Returns:
        list: List of color tuples
    """
    if isinstance(color, tuple):
        return [color] * num_bboxes
    elif isinstance(color, list):
        if len(color) == num_bboxes:
            return color
        else:
            # Repeat colors if needed
            return [color[i % len(color)] for i in range(num_bboxes)]
    else:
        raise ValueError(f"Invalid color type: {type(color)}")


def filter_predictions_by_score(pred_data, score_threshold=0.3):
    """Filter predictions by score threshold.
    
    Args:
        pred_data: Dictionary with 'bboxes_3d', 'labels_3d', 'scores_3d'
        score_threshold: Minimum score to keep (default: 0.3)
        
    Returns:
        dict: Filtered prediction data with same structure
    """
    if len(pred_data['scores_3d']) == 0:
        return pred_data
    
    # Convert to numpy for easier filtering
    scores = np.array(pred_data['scores_3d'])
    mask = scores >= score_threshold
    
    if not isinstance(pred_data['bboxes_3d'], (LiDARInstance3DBoxes, CameraInstance3DBoxes)):
        raise ValueError("bboxes_3d must be LiDARInstance3DBoxes or CameraInstance3DBoxes")
    
    # Filter bboxes
    bboxes_tensor = pred_data['bboxes_3d'].tensor[mask]
    box_type = pred_data['box_type_3d']
    
    if box_type == 'LiDAR':
        filtered_bboxes = LiDARInstance3DBoxes(bboxes_tensor)
    else:
        filtered_bboxes = CameraInstance3DBoxes(bboxes_tensor)
    
    # Filter labels and scores
    filtered_labels = [pred_data['labels_3d'][i] for i in range(len(mask)) if mask[i]]
    filtered_scores = scores[mask].tolist()
    
    return {
        'bboxes_3d': filtered_bboxes,
        'labels_3d': filtered_labels,
        'scores_3d': filtered_scores,
        'box_type_3d': box_type
    }


# ============================================================================
# Path Configuration
# ============================================================================

def get_dataset_paths(base_dataset_dir=None, sequence_id='0000', phase_type='training'):
    """Get standardized dataset directory paths.
    
    Args:
        base_dataset_dir: Base directory for datasets (default: '/home/hasan/workspace/datasets/multimodal/tracking')
        sequence_id: Sequence ID (default: '0000')
        phase_type: Phase type, e.g., 'training' or 'testing' (default: 'training')
        
    Returns:
        dict: Dictionary containing 'base', 'new_tracking', 'org_tracking', 'sequence_id', 'phase_type'
    """
    if base_dataset_dir is None:
        base_dataset_dir = '/home/hasan/workspace/datasets/multimodal/tracking'
    
    return {
        'base': base_dataset_dir,
        'new_tracking': f'{base_dataset_dir}/newTracking/{sequence_id}',
        'org_tracking': f'{base_dataset_dir}/orgTracking',
        'sequence_id': sequence_id,
        'phase_type': phase_type
    }


def calculate_delay_params(delay, delay_mode, idx):
    """Calculate delay parameters for a given delay value and mode.
    
    Args:
        delay: Current delay value (integer)
        delay_mode: Delay mode - 'lidar', 'image', or 'both'
        idx: Base frame index
        
    Returns:
        dict: Dictionary containing 'delay_image', 'delay_lidar', 'image_file', 'lidar_file'
        
    Raises:
        ValueError: If delay_mode is invalid
    """
    if delay_mode == 'lidar':
        # Delay LiDAR: image_file stays constant, lidar_file decreases
        delay_image = 0
        delay_lidar = delay
        image_file = idx
        lidar_file = max(0, idx - delay_lidar)
    elif delay_mode == 'image':
        # Delay Image: lidar_file stays constant, image_file decreases
        delay_image = delay
        delay_lidar = 0
        image_file = max(0, idx - delay_image)
        lidar_file = idx
    elif delay_mode == 'both':
        # Delay both: both decrease
        delay_image = delay
        delay_lidar = delay
        image_file = max(0, idx - delay_image)
        lidar_file = max(0, idx - delay_lidar)
    else:
        raise ValueError(f"Invalid delay_mode: {delay_mode}. Must be 'lidar', 'image', or 'both'")
    
    return {
        'delay_image': delay_image,
        'delay_lidar': delay_lidar,
        'image_file': image_file,
        'lidar_file': lidar_file
    }


def generate_prediction_file_paths(new_tracking_dir, phase_type, idx, delay_mode='lidar', max_delay=1, delays=None):
    """Generate list of prediction file paths based on delay configuration.
    
    Args:
        new_tracking_dir: Path to newTracking directory for the sequence
        phase_type: Phase type, e.g., 'training' or 'testing'
        idx: Base frame index
        delay_mode: Delay mode - 'lidar', 'image', or 'both' (default: 'lidar')
        max_delay: Maximum delay to loop over (default: 1). Ignored if `delays` is provided.
        delays: Optional iterable of explicit delay values to use instead of 0..max_delay
        
    Returns:
        list: List of prediction file paths
        
    Example:
        >>> paths = generate_prediction_file_paths(
        ...     '/path/to/newTracking/0000',
        ...     'training',
        ...     idx=5,
        ...     delay_mode='lidar',
        ...     max_delay=3
        ... )
        >>> # Returns paths like:
        >>> # ['/path/to/newTracking/0000/training/vis_results/preds/0_0_000005_000005.json',
        >>> #  '/path/to/newTracking/0000/training/vis_results/preds/0_1_000005_000004.json',
        >>> #  ...]
    """
    prediction_files = []

    # If an explicit list of delays is provided, use it; otherwise fall back to range(max_delay + 1)
    if delays is not None:
        delay_values = list(delays)
    else:
        delay_values = list(range(max_delay + 1))
    
    for delay in delay_values:
        params = calculate_delay_params(delay, delay_mode, idx)
        pred_file = (
            f'{new_tracking_dir}/{phase_type}/vis_results/preds/'
            f'{params["delay_image"]}_{params["delay_lidar"]}_'
            f'{params["image_file"]:06d}_{params["lidar_file"]:06d}.json'
        )
        prediction_files.append(pred_file)
    
    return prediction_files


# ============================================================================
# Visualization Configuration
# ============================================================================

# Default wait_time for visualizations
# -1: Keep window open indefinitely (for interactive inspection)
# 0.01: Auto-close window after 0.01 seconds (for batch processing)
VISUALIZATION_WAIT_TIME = -1  # Change this to 0.01 for batch processing


def get_visualizer_with_config(alpha=0):
    """Get a Det3DLocalVisualizer instance with project-level configuration.
    
    This wrapper ensures consistent visualization settings across the project.
    
    Args:
        alpha: Transparency level for visualizations (default: 0)
        
    Returns:
        Det3DLocalVisualizer: Configured visualizer instance
    """
    from mmdet3d.visualization import Det3DLocalVisualizer
    return Det3DLocalVisualizer(alpha=alpha)


def show_visualization(visualizer, wait_time=None, **kwargs):
    """Show visualization with project-level wait_time configuration.
    
    This wrapper ensures consistent wait_time settings across the project.
    If wait_time is not specified, uses VISUALIZATION_WAIT_TIME from config.
    
    Args:
        visualizer: Det3DLocalVisualizer instance
        wait_time: Optional wait time override. If None, uses VISUALIZATION_WAIT_TIME
        **kwargs: Additional arguments to pass to visualizer.show()
    """
    if wait_time is None:
        wait_time = VISUALIZATION_WAIT_TIME
    
    # Handle different show() method signatures
    if 'save_path' in kwargs:
        visualizer.show(save_path=kwargs['save_path'], wait_time=wait_time)
    else:
        visualizer.show(wait_time=wait_time)

