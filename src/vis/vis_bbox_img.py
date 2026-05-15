"""
Visualize 3D bounding boxes projected on camera images.
Supports ground truth and prediction bounding boxes.
"""
import os
import glob
import re
from pathlib import Path
import mmcv
import numpy as np
import hydra
from omegaconf import OmegaConf
from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.structures import CameraInstance3DBoxes

try:
    from .vis_utils import (
        load_gt_from_pkl,
        load_prediction_from_json,
        lidar_boxes_to_camera,
        GT_COLOR,
        get_bbox_colors,
        get_delay_color,
        extract_delay_from_filename,
        get_dataset_paths,
        generate_prediction_file_paths
    )
except ImportError:
    # Fallback for direct script execution
    from vis_utils import (
        load_gt_from_pkl,
        load_prediction_from_json,
        lidar_boxes_to_camera,
        GT_COLOR,
        get_bbox_colors,
        get_delay_color,
        extract_delay_from_filename,
        get_dataset_paths,
        generate_prediction_file_paths
    )


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(args: OmegaConf) -> None:
    """
    Main function to visualize 3D bounding boxes on camera images.
    
    Args:
        args: Hydra config object containing dataset and visualization configuration
    """
    # Get configuration from Hydra config
    dataset_cfg = args.dataset
    vis_cfg = dataset_cfg.visualization
    
    # Dataset configuration
    SEQUENCE_ID = vis_cfg.sequence_id
    PHASE_TYPE = dataset_cfg.phaseType
    
    # Get starting frame index from vis_start config based on sequence_id
    # YAML parses numeric keys as strings when quoted, so we need to handle both string and int keys
    vis_start_config = OmegaConf.to_container(getattr(dataset_cfg, 'vis_start', {}), resolve=True) or {}
    trip_id_int = int(SEQUENCE_ID)
    start_frame_value = (
        vis_start_config.get(SEQUENCE_ID) or  # Try string key "0001"
        vis_start_config.get(trip_id_int) or  # Try integer key 1
        vis_start_config.get(str(trip_id_int))  # Try string key "1"
    )
    
    # Delay configuration (needed before calculating idx)
    delay_mode = vis_cfg.get('delay_mode', 'lidar')  # Default to 'lidar' if not specified

    # Support either an explicit delay list or a max_delay range for backward compatibility
    delay_list_cfg = vis_cfg.get('delay_list', None)
    if delay_list_cfg is not None:
        # Convert to a simple Python list (OmegaConf containers → list)
        delays = list(delay_list_cfg)
        max_delay = max(delays) if len(delays) > 0 else 0
    else:
        max_delay = vis_cfg.max_delay
        delays = list(range(max_delay + 1))
    
    # Get dataset paths early to check for existing prediction files
    base_dataset_dir = os.path.dirname(dataset_cfg.srcDirRoot.rstrip('/'))
    paths = get_dataset_paths(
        base_dataset_dir=base_dataset_dir,
        sequence_id=SEQUENCE_ID,
        phase_type=dataset_cfg.phaseType
    )
    NEW_TRACKING_DIR = paths['new_tracking']
    pred_dir = Path(f'{NEW_TRACKING_DIR}/{dataset_cfg.phaseType}/vis_results/preds')
    
    if start_frame_value is not None:
        # Convert to int (handles both string "000360" and integer values)
        start_idx = int(start_frame_value) - 2
        
        # Try to find existing prediction files to determine the correct base index
        # Prediction files are named like: {delay_image}_{delay_lidar}_{image_file:06d}_{lidar_file:06d}.json
        # For lidar delay mode with delay=0: image_file = idx, lidar_file = idx
        # So we look for files matching: 0_0_{idx:06d}_{idx:06d}.json
        
        # First, try the starting index directly (prediction files might be generated from start_idx)
        idx = start_idx
        
        # Check if prediction files exist for this idx
        if pred_dir.exists():
            # Try to find existing prediction files to infer the correct base index
            # Look for any prediction files: {delay_image}_{delay_lidar}_{image_file:06d}_{lidar_file:06d}.json
            # Also check for .png files as they might indicate the pattern
            pattern_json = f'{pred_dir}/*_*_*_*.json'
            pattern_png = f'{pred_dir}/*_*_*_*.png'
            existing_files = glob.glob(pattern_json) + glob.glob(pattern_png)
            
            if existing_files:
                # Extract image_file from existing files and group by image_file value
                # File format: {delay_image}_{delay_lidar}_{image_file:06d}_{lidar_file:06d}.{ext}
                index_counts = {}  # {image_file: list of (delay_image, delay_lidar, lidar_file)}
                for file_path in existing_files:
                    match = re.search(r'(\d+)_(\d+)_(\d{6})_(\d{6})\.(json|png)', file_path)
                    if match:
                        delay_image = int(match.group(1))
                        delay_lidar = int(match.group(2))
                        image_file = int(match.group(3))
                        lidar_file = int(match.group(4))
                        
                        # Check if this idx makes sense (should be >= start_idx)
                        if image_file >= start_idx:
                            if image_file not in index_counts:
                                index_counts[image_file] = []
                            index_counts[image_file].append((delay_image, delay_lidar, lidar_file))
                
                if index_counts:
                    # Find the index that has files for the most delays (especially max_delay)
                    # Prefer indices that have files matching the expected delay pattern
                    best_idx = None
                    best_score = -1
                    best_delays = set()
                    
                    for candidate_idx in sorted(index_counts.keys(), reverse=True):
                        # Count how many delay values we have files for
                        files_for_idx = index_counts[candidate_idx]
                        delays_found = set()
                        for delay_img, delay_lid, lid_file in files_for_idx:
                            if delay_mode == 'lidar':
                                delays_found.add(delay_lid)
                            elif delay_mode == 'image':
                                delays_found.add(delay_img)
                            elif delay_mode == 'both':
                                delays_found.add(max(delay_img, delay_lid))
                        
                        # Score: prefer indices that have files for max_delay
                        score = len(delays_found)
                        if max_delay in delays_found:
                            score += 10  # Bonus for having max_delay
                        # Also prefer indices closer to start_idx + max_delay
                        expected_idx = start_idx + max_delay
                        if abs(candidate_idx - expected_idx) <= 10:  # Within 10 frames
                            score += 5
                        
                        if score > best_score:
                            best_score = score
                            best_idx = candidate_idx
                            best_delays = delays_found
                    
                    if best_idx is not None:
                        idx = best_idx
                        print(f"Found existing prediction files, using base index: {idx} (found {len(index_counts[idx])} files for this index, covering delays: {sorted(best_delays)})")
                    else:
                        # Fallback to maximum index
                        idx = max(index_counts.keys())
                        print(f"Found existing prediction files, using base index: {idx} (maximum found)")
        
        print(f"Using starting frame index from vis_start config: {start_idx} (for sequence {SEQUENCE_ID})")
        print(f"Using visualization index: {idx}")
    else:
        # Fallback to visualization.idx if not found in vis_start
        idx = vis_cfg.get('idx', 0)
        print(f"Starting frame index not found in vis_start for sequence {SEQUENCE_ID}, using visualization.idx: {idx}")
    
    # Optional configuration
    score_threshold = vis_cfg.get('score_threshold', 0.3)
    if score_threshold == 'null' or score_threshold is None:
        score_threshold = None
    
    save_path = vis_cfg.get('save_path', None)
    if save_path == 'null' or save_path is None:
        save_path = None
    elif save_path and (os.path.isdir(save_path) or save_path.endswith('/')):
        # If save_path is a directory, construct filename from visualization attributes
        # Normalize the directory path (remove trailing slash if present)
        save_dir = save_path.rstrip('/')
        delay_list_str = '_'.join(map(str, delays)) if delays else '0'
        filename = f"img_seq_{SEQUENCE_ID}_{delay_mode}_delays_{delay_list_str}_frame_{idx:06d}.png"
        save_path = os.path.join(save_dir, filename)
    
    # Dataset paths already retrieved above, just get the tracking dirs
    ORG_TRACKING_DIR = paths['org_tracking']
    
    # Generate file paths
    file_idx = f'{idx:06d}'
    gt_pkl_path = f'{NEW_TRACKING_DIR}/kitti_infos_train_indv/kitti_infos_train_{file_idx}.pkl'
    image_path = f'{ORG_TRACKING_DIR}/{PHASE_TYPE}/image_2/{SEQUENCE_ID}/{file_idx}.png'
    
    # Generate prediction file paths
    prediction_files = generate_prediction_file_paths(
        NEW_TRACKING_DIR,
        PHASE_TYPE,
        idx,
        delay_mode=delay_mode,
        max_delay=max_delay,
        delays=delays,
    )

    # ============================================================================
    # Load Ground Truth
    # ============================================================================
    print(f"Loading ground truth from: {gt_pkl_path}")
    gt_data = load_gt_from_pkl(gt_pkl_path, file_idx)
    gt_bboxes_3d = gt_data['bboxes_3d']  # Already CameraInstance3DBoxes
    cam2img = gt_data['cam2img']
    input_meta = {'cam2img': cam2img}
    
    print(f"Loaded {len(gt_bboxes_3d)} ground truth bounding boxes")
    
    
    # ============================================================================
    # Load Predictions (if any)
    # ============================================================================
    predictions = []
    for pred_file in prediction_files:
        # Check if file exists before trying to load
        if not os.path.exists(pred_file):
            print(f"Warning: Prediction file not found: {pred_file}")
            print(f"  Skipping this prediction file. Available files in directory:")
            pred_dir = os.path.dirname(pred_file)
            if os.path.exists(pred_dir):
                available_files = sorted(glob.glob(f'{pred_dir}/*.json'))[:10]  # Show first 10
                for avail_file in available_files:
                    print(f"    {os.path.basename(avail_file)}")
            continue
        
        print(f"Loading prediction from: {pred_file}")
        pred_data = load_prediction_from_json(pred_file, score_threshold=score_threshold)
        
        if score_threshold is not None:
            print(f"  Filtered predictions with score >= {score_threshold}")
        
        # Convert to camera coordinates if needed
        if pred_data['box_type_3d'] == 'LiDAR':
            pred_bboxes_cam = lidar_boxes_to_camera(pred_data['bboxes_3d'])
        else:
            pred_bboxes_cam = pred_data['bboxes_3d']
        
        # Extract delay from filename for color assignment
        delay_value = extract_delay_from_filename(pred_file, delay_mode=delay_mode)
        
        predictions.append({
            'bboxes_3d': pred_bboxes_cam,
            'labels_3d': pred_data['labels_3d'],
            'scores_3d': pred_data['scores_3d'],
            'name': pred_file,  # Will use for labeling later
            'delay': delay_value  # Store delay for color assignment
        })
        print(f"  Loaded {len(pred_bboxes_cam)} predicted bounding boxes (after filtering) [Delay: {delay_value}]")
    
    
    # ============================================================================
    # Load and Setup Image
    # ============================================================================
    print(f"Loading image from: {image_path}")
    img = mmcv.imread(image_path)
    img = mmcv.imconvert(img, 'bgr', 'rgb')
    
    visualizer = Det3DLocalVisualizer()
    visualizer.set_image(img)
    
    
    # ============================================================================
    # Draw Ground Truth Bounding Boxes
    # ============================================================================
    if len(gt_bboxes_3d) > 0:
        gt_colors = get_bbox_colors(len(gt_bboxes_3d), GT_COLOR)
        # face_colors='none' removes the fill/shading to avoid blurring the image
        visualizer.draw_proj_bboxes_3d(gt_bboxes_3d, input_meta, edge_colors=gt_colors, face_colors='none', alpha=0)
        print(f"Drew {len(gt_bboxes_3d)} ground truth bounding boxes (green)")
    
    
    # ============================================================================
    # Draw Prediction Bounding Boxes
    # ============================================================================
    # Use delay-based colors: delay 0 (greenish) -> delay 5 (red)
    for pred in predictions:
        if len(pred['bboxes_3d']) > 0:
            # Get color based on delay value
            delay_value = pred.get('delay', 0)
            delay_color = get_delay_color(delay_value)
            pred_colors = get_bbox_colors(len(pred['bboxes_3d']), delay_color)
            # face_colors='none' removes the fill/shading to avoid blurring the image
            visualizer.draw_proj_bboxes_3d(pred['bboxes_3d'], input_meta, edge_colors=pred_colors, face_colors='none', alpha=0)
            print(f"Drew {len(pred['bboxes_3d'])} predicted bounding boxes [Delay: {delay_value}, Color: {delay_color}]")
    
    
    # ============================================================================
    # Display Visualization and Save Snapshot
    # ============================================================================
    if save_path is not None:
        # If saving, skip showing to avoid blocking (we'll save directly)
        # Get the drawn image (RGB format)
        drawn_img = visualizer.get_image()
        
        # Convert RGB to BGR for mmcv.imwrite (mmcv expects BGR format)
        drawn_img_bgr = drawn_img[..., ::-1]
        
        # Ensure directory exists
        save_dir = os.path.dirname(save_path)
        if save_dir:  # Only create directory if path contains a directory
            os.makedirs(save_dir, exist_ok=True)
        
        # Ensure file extension
        if not (save_path.endswith('.png') or save_path.endswith('.jpg') or save_path.endswith('.jpeg')):
            save_path += '.png'
        
        # Save the image
        mmcv.imwrite(drawn_img_bgr, save_path)
        print(f"Snapshot saved to: {save_path}")
    else:
        # Only show if not saving (for interactive viewing)
        # Auto-close window after 0.01 seconds (for batch processing)
        visualizer.show(wait_time=0.01)


if __name__ == '__main__':
    main()