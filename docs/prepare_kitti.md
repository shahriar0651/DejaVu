# KITTI Dataset Preparation

## Overview

The KITTI 3D object detection dataset is a widely used multimodal dataset for autonomous driving. However, the standard KITTI detection dataset does not contain sequential samples; rather, it contains instances from random scenes. For temporal misalignment attack evaluation, we need sequential data, so we use the **KITTI Tracking dataset** and convert it to the detection format.

For more details about the KITTI detection format, see the [official KITTI documentation](https://github.com/bostondiditeam/kitti/blob/master/resources/devkit_object/readme.txt).

## Quick Start (Automated)

**Recommended**: Use the provided shell script to automate the entire dataset preparation process:

```bash
# Run all steps automatically
./script/prepare_kitti.sh all

# Or run individual steps
./script/prepare_kitti.sh 1        # Download dataset
./script/prepare_kitti.sh 3        # Convert to detection format
./script/prepare_kitti.sh 4        # Create info files for all sequences
./script/prepare_kitti.sh 4 0000  # Create info files for sequence 0000 only

# Show help
./script/prepare_kitti.sh help
```

The script handles all the steps below automatically, including directory renaming, error handling, and processing all sequences. See the [script documentation](#automated-script-usage) section for more details.

## Dataset Preparation

**Note**: The following sections provide manual step-by-step instructions. If you prefer automation, use the [shell script](#quick-start-automated) above.

### Step 1: Download KITTI Tracking Dataset

**Automated**: Run `./script/prepare_kitti.sh 1` or `./script/prepare_kitti.sh all`

**Manual**: Download the KITTI Tracking dataset from the official source:

```bash
mkdir -p ~/workspace/datasets/multimodal/tracking/orgTracking
cd ~/workspace/datasets/multimodal/tracking/orgTracking

# Download all required files
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_image_2.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_velodyne.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_label_2.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_calib.zip

# Extract all zip files
unzip "*.zip"
```

**Important**: After extraction, rename the `image_02` and `label_02` directories to `image_2` and `label_2` respectively to match the expected naming convention:

```bash
# Rename image_02 to image_2 and label_02 to label_2 in the training directory
cd ~/workspace/datasets/multimodal/tracking/orgTracking/training
mv image_02 image_2
mv label_02 label_2
```

Or using the full path:

```bash
mv ~/workspace/datasets/multimodal/tracking/orgTracking/training/image_02 \
   ~/workspace/datasets/multimodal/tracking/orgTracking/training/image_2

mv ~/workspace/datasets/multimodal/tracking/orgTracking/testing/image_02 \
   ~/workspace/datasets/multimodal/tracking/orgTracking/testing/image_2

mv ~/workspace/datasets/multimodal/tracking/orgTracking/training/label_02 \
   ~/workspace/datasets/multimodal/tracking/orgTracking/training/label_2
```

### Step 2: Update Configuration

**Note**: This step must be done manually. The shell script will prompt you to update the configuration.

Edit the configuration file `config/dataset/kitti.yaml` to set the correct dataset paths. Update the following paths to match your system:

- `org_tracking_path`: Path to the original KITTI tracking dataset
- `converted_detection_path`: Path where the converted detection dataset will be stored
- `poisoned_detection_path`: Path where the poisoned (temporally misaligned) dataset will be stored

### Step 3: Convert to Detection Format

**Automated**: Run `./script/prepare_kitti.sh 3` or `./script/prepare_kitti.sh all`

**Manual**: Convert the KITTI Tracking dataset to the detection format required by MMDetection3D:

```bash
cd src
python 0_convert_kitti_tracking.py
```

This script will:
- Process the tracking sequences
- Convert annotations to detection format
- Generate the required info files (`.pkl` files) for MMDetection3D
- Organize the data in the expected directory structure


### Step 4: Create Dataset Info Files

**Automated**: Run `./script/prepare_kitti.sh 4` (all sequences) or `./script/prepare_kitti.sh 4 0000` (single sequence)

**Manual**: Generate the required info files for MMDetection3D. Since the converted dataset contains multiple sequence folders (e.g., `0000`, `0001`, `0002`, etc.), you need to process each sequence separately.

**Option 1: Process All Sequences Automatically**

Loop over each sequence folder in the converted dataset and generate info files:
Run this from `DejaVu` home: 

```bash
# Get absolute path to mmdetection3d directory
MMDET3D_DIR="$(pwd)/mmdetection3d"
DATA_DIR="$MMDET3D_DIR/data"
TRACKING_DIR="/home/hasan/workspace/datasets/multimodal/tracking/newTracking"

# Navigate to mmdetection3d
cd "$MMDET3D_DIR" || exit 1

# Loop over each sequence folder in newTracking
for seq_folder in "$TRACKING_DIR"/*/; do
    # Check if the folder exists and is a directory
    if [ ! -d "$seq_folder" ]; then
        echo "Skipping non-directory: $seq_folder"
        continue
    fi
    
    # Get the sequence name (e.g., 0000, 0001, etc.)
    seq_name=$(basename "$seq_folder")
    echo "========================================="
    echo "Processing sequence: $seq_name"
    echo "========================================="
    
    # Navigate to MMDetection3D data directory
    cd "$DATA_DIR" || {
        echo "Error: Cannot access $DATA_DIR"
        continue
    }
    
    # Remove existing symlink if it exists
    rm -f kitti
    
    # Create symbolic link to the current sequence folder
    ln -s "$seq_folder" kitti || {
        echo "Error: Failed to create symlink for sequence $seq_name"
        continue
    }
    
    # Go back to mmdetection3d root
    cd "$MMDET3D_DIR" || {
        echo "Error: Cannot return to $MMDET3D_DIR"
        continue
    }
    
    # Generate info files for this sequence
    if python tools/create_data.py kitti \
        --root-path ./data/kitti \
        --out-dir ./data/kitti \
        --extra-tag kitti; then
        echo "✓ Successfully completed processing sequence: $seq_name"
    else
        echo "✗ Error processing sequence: $seq_name (continuing to next sequence)"
    fi
    echo ""
done

echo "========================================="
echo "Finished processing all sequences"
echo "========================================="
```

**Option 2: Process a Single Sequence**

If you want to process a specific sequence (e.g., sequence `0000`):

```bash
cd mmdetection3d/data

# Remove existing symlink if it exists
rm -f kitti

# Create symbolic link to the specific sequence
ln -s ~/workspace/datasets/multimodal/tracking/newTracking/0000 kitti

# Go back to mmdetection3d root
cd ..

# Generate info files
python tools/create_data.py kitti \
    --root-path ./data/kitti \
    --out-dir ./data/kitti \
    --extra-tag kitti
```

This script will:
- Process each KITTI sequence folder
- Generate `.pkl` info files required by MMDetection3D for each sequence
- Create the necessary data structure in each sequence folder

**Note**: The info files will be generated within each sequence folder (e.g., `newTracking/0000/kitti_infos_train.pkl`).

## Automated Script Usage

The `script/prepare_kitti.sh` script automates all the preparation steps. Here's how to use it:

### Basic Usage

```bash
# From the DejaVu root directory
./script/prepare_kitti.sh [step] [options]
```

### Available Steps

- `1` or `download` - Download KITTI Tracking dataset
- `2` or `config` - Show configuration reminder (manual step)
- `3` or `convert` - Convert tracking to detection format
- `4` or `info` - Create dataset info files
- `all` - Run all steps sequentially (default)

### Examples

```bash
# Run all steps
./script/prepare_kitti.sh all

# Download dataset only
./script/prepare_kitti.sh 1

# Convert dataset only
./script/prepare_kitti.sh 3

# Create info files for all sequences
./script/prepare_kitti.sh 4

# Create info files for a specific sequence
./script/prepare_kitti.sh 4 0000

# Show help
./script/prepare_kitti.sh help
```

### Environment Variables

You can customize paths using environment variables:

```bash
export ORG_TRACKING_DIR="$HOME/workspace/datasets/multimodal/tracking/orgTracking"
export CONVERTED_TRACKING_DIR="$HOME/workspace/datasets/multimodal/tracking/newTracking"
export DEJAVU_ROOT="$(pwd)"
export MMDET3D_DIR="$DEJAVU_ROOT/mmdetection3d"

./script/prepare_kitti.sh all
```

### Features

- ✅ Automatic directory renaming (`image_02` → `image_2`, `label_02` → `label_2`)
- ✅ Error handling with colored output
- ✅ Processes all sequences automatically
- ✅ Continues processing even if one sequence fails
- ✅ Progress indicators for each step

## Running Demos

### MVXNet Demo

#### Using MMDetection3D Demo Data

Test with the included demo data:

```bash
cd mmdetection3d
python demo/multi_modality_demo.py \
    demo/data/kitti/000008.bin \
    demo/data/kitti/000008.png \
    demo/data/kitti/000008.pkl \
    configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py \
    checkpoints/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class_20210831_060805-83442923.pth \
    --show
```

#### Using Converted KITTI Tracking Dataset

**Alert!!!**: You can only run this once you are done with `mm_poison_and_evaluate_detect_kitti_dataset.py`, it depends on some files that are generated along that script.

Run demo on your converted dataset:

```bash
cd mmdetection3d
python demo/multi_modality_demo.py \
    ../../datasets/multimodal/tracking/newTracking/0000/training/velodyne_reduced/000000.bin \
    ../../datasets/multimodal/tracking/newTracking/0000/training/image_2/000000.png \
    ../../datasets/multimodal/tracking/newTracking/0000/kitti_infos_train_indv/kitti_infos_train_000000.pkl \
    configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py \
    checkpoints/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class_20210831_060805-83442923.pth \
    --show
```

**Note**: Adjust the paths according to your dataset location and sequence number.

## Evaluation

### MVXNet Evaluation

Evaluate MVXNet on the KITTI dataset:

```bash
cd mmdetection3d

# Generate JSON results
python tools/test.py \
    configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py \
    checkpoints/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class-8963258a.pth \
    --cfg-options 'test_evaluator.jsonfile_prefix=./mvxnet_kitti_results'

# Generate PKL results
python tools/test.py \
    configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py \
    checkpoints/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class-8963258a.pth \
    --cfg-options 'test_evaluator.pklfile_prefix=./mvxnet_kitti_results'
```

### Expected Results

Example evaluation results (baseline performance):

```
Overall AP40@easy, moderate, hard:
bbox AP40: 69.1154, 69.9577, 76.0771
bev  AP40: 68.5164, 68.4788, 72.6443
3d   AP40: 68.4316, 68.3824, 70.5076
```

Example results with temporal misalignment attack (Image Delay: 5 frames):

```
Overall AP40@easy, moderate, hard:
bbox AP40: 4.4296, 4.4759, 4.8908
bev  AP40: 0.8126, 0.8126, 0.8126
3d   AP40: 0.0901, 0.0901, 0.0901
```

## Notes

- The conversion script (`0_convert_kitti_tracking.py`) processes sequences individually and creates separate info files for each sequence.
- Make sure you have sufficient disk space for the converted dataset (approximately 2-3x the original tracking dataset size).
- The evaluation scripts expect the dataset to be in the MMDetection3D-compatible format generated by the conversion script.

## Troubleshooting

**Issue**: Missing `DontCare` class in annotations
- The conversion script should handle this automatically. If you encounter issues, check that the label files are properly formatted.

**Issue**: Path errors during conversion
- Verify that all paths in `config/dataset/kitti.yaml` are correct and accessible.
- Ensure the original tracking dataset is fully extracted and the `image_02` directory has been renamed to `image_2`.
