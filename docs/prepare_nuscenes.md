# nuScenes Dataset Preparation

## Overview

The nuScenes dataset is a large-scale multimodal autonomous driving dataset that includes sequential samples with synchronized camera and LiDAR data. This makes it ideal for evaluating temporal misalignment attacks. Unlike KITTI, nuScenes already provides sequential data in the detection format, so no conversion is needed.

For more details about the nuScenes dataset format, see the [MMDetection3D nuScenes documentation](https://mmdetection3d.readthedocs.io/en/v0.18.1/datasets/nuscenes_det.html) and the [official nuScenes website](https://www.nuscenes.org/).

## Dataset Preparation

### Step 1: Download nuScenes Dataset

Download the nuScenes dataset following the official instructions:

1. **Create an account** on the [nuScenes website](https://www.nuscenes.org/download)
2. **Request access** to the dataset (requires approval)
3. **Download the dataset** to your local machine

**Recommended directory structure**:

```bash
mkdir -p ~/workspace/datasets/multimodal/detection/nuscenes/
cd ~/workspace/datasets/multimodal/detection/nuscenes/
```

**Download links** (after approval):
- Maps: `v1.0-trainval_meta.tgz`
- Trainval: `v1.0-trainval01_blobs.tgz` through `v1.0-trainval10_blobs.tgz`
- Test: `v1.0-test_blobs.tgz` (optional, for submission)

**Extract the dataset**:

```bash
# Extract all downloaded files
tar -xzf v1.0-trainval_meta.tgz
tar -xzf v1.0-trainval01_blobs.tgz
# ... extract remaining blobs files
```

**Note**: The dataset is large (~300GB). Ensure you have sufficient disk space and bandwidth.

### Step 2: Update Configuration

Edit the configuration file `config/dataset/nuscenes.yaml` to set the correct dataset paths:

- `data_root`: Root path to the nuScenes dataset (should contain `v1.0-trainval/` directory)
- `version`: Dataset version (typically `v1.0-trainval`)
- `poisoned_data_root`: Path where the poisoned (temporally misaligned) dataset will be stored

### Step 3: Create Dataset Info Files

Generate the required info files for MMDetection3D:

```bash
cd mmdetection3d
python tools/create_data.py nuscenes \
    --root-path ~/workspace/datasets/multimodal/detection/nuscenes/v1.0-trainval/ \
    --out-dir ~/workspace/datasets/multimodal/detection/nuscenes/v1.0-trainval/ \
    --extra-tag nuscenes
```

This script will:
- Process the nuScenes dataset
- Generate `.pkl` info files required by MMDetection3D
- Create the necessary data structure

**Note**: This process may take 30-60 minutes depending on your system.

### Step 4: Link Dataset to MMDetection3D (Optional)

If needed, create a symbolic link from MMDetection3D's data directory:

```bash
cd mmdetection3d/data
ln -s ~/workspace/datasets/multimodal/detection/nuscenes/v1.0-trainval nuscenes
```

### Step 5: Install ImageMagick (Optional)

For some visualization and processing tasks, ImageMagick may be required:

```bash
sudo apt install imagemagick
```

## Running Demos

### BEVFusion Multimodal Demo

#### Using MMDetection3D Demo Data

Test with the included demo data:

```bash
cd mmdetection3d
python projects/BEVFusion/demo/multi_modality_demo.py \
    demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin \
    demo/data/nuscenes/ \
    demo/data/nuscenes/n015-2018-07-24-11-22-45+0800.pkl \
    projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
    checkpoints/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af_v2.pth \
    --cam-type all \
    --score-thr 0.2 \
    --show
```

#### Using Demo Dataset

```bash
cd mmdetection3d
python demo/multi_modality_demo.py \
    demo/data/nuscenes-demo/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin \
    demo/data/nuscenes-demo/ \
    demo/data/nuscenes-demo/n015-2018-07-24-11-22-45+0800.pkl \
    projects/BEVFusion/configs/bevfusion_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
    checkpoints/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth \
    --cam-type all \
    --score-thr 0.2 \
    --show
```

### Monocular Detection Demo

Run monocular (camera-only) detection:

```bash
cd mmdetection3d
python demo/mono_det_demo.py \
    demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525.jpg \
    demo/data/nuscenes/n015-2018-07-24-11-22-45+0800.pkl \
    configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune.py \
    checkpoints/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth \
    --show \
    --cam-type CAM_BACK
```

## Evaluation

### PointPillars Evaluation

Evaluate PointPillars on nuScenes:

```bash
cd mmdetection3d
bash ./tools/dist_test.sh \
    configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py \
    checkpoints/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20200620_230405-2fa62f3d.pth \
    2
```

**Expected Results** (baseline PointPillars):

```
mAP: 0.3197
mATE: 0.7595
mASE: 0.2700
mAOE: 0.4918
mAVE: 1.3307
mAAE: 0.1724
NDS: 0.3905
Eval time: 170.8s

Per-class results:
Object Class            AP      ATE     ASE     AOE     AVE     AAE
car                     0.503   0.577   0.152  0.111   2.096   0.136
truck                   0.223   0.857   0.224  0.220   1.389   0.179
bus                     0.294   0.855   0.204  0.190   2.689   0.283
trailer                  0.081   1.094   0.243  0.553   0.742   0.167
construction_vehicle     0.058   1.017   0.450  1.019   0.137   0.341
pedestrian               0.392   0.687   0.284  0.694   0.876   0.158
motorcycle               0.317   0.737   0.265  0.580   2.033   0.104
bicycle                  0.308   0.704   0.299  0.892   0.683   0.010
traffic_cone             0.555   0.486   0.309  nan     nan     nan
barrier                  0.466   0.581   0.269  0.169   nan     nan
```

### BEVFusion Evaluation

Evaluate BEVFusion (multimodal) on nuScenes:

```bash
cd mmdetection3d
bash tools/dist_test.sh \
    projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
    checkpoints/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af_v2.pth \
    2 \
    --cfg-options 'test_evaluator.jsonfile_prefix=results_dir/bev-nus/results_eval'
```

**Expected Results** (baseline BEVFusion):

```
mAP: 0.6833
mATE: 0.2787
mASE: 0.2547
mAOE: 0.2981
mAVE: 0.2790
mAAE: 0.1874
NDS: 0.7119
Eval time: 83.5s

Per-class results:
Object Class            AP      ATE     ASE     AOE     AVE     AAE
car                     0.893   0.170   0.150   0.063   0.284   0.185
truck                   0.640   0.319   0.181   0.085   0.255   0.223
bus                     0.765   0.332   0.184   0.054   0.471   0.255
trailer                  0.477   0.505   0.210   0.624   0.191   0.171
construction_vehicle     0.290   0.689   0.423   0.842   0.131   0.311
pedestrian               0.879   0.130   0.292   0.393   0.227   0.104
motorcycle               0.759   0.183   0.238   0.261   0.466   0.241
bicycle                  0.625   0.155   0.257   0.302   0.207   0.009
traffic_cone             0.789   0.120   0.327   nan     nan     nan
barrier                  0.716   0.184   0.285   0.059   nan     nan
```

## Notes

- The nuScenes dataset is significantly larger than KITTI (~300GB vs ~80GB).
- The dataset includes 10 camera views and LiDAR data with precise timestamps.
- Evaluation metrics include mAP (mean Average Precision), NDS (nuScenes Detection Score), and various error metrics (ATE, ASE, AOE, AVE, AAE).
- The `create_data.py` script processes the dataset and may take considerable time.

## Troubleshooting

**Issue**: Dataset download fails or is slow
- The dataset is very large. Use a stable internet connection and consider downloading during off-peak hours.
- You may need to request access multiple times if your initial request is not approved.

**Issue**: `create_data.py` fails or takes too long
- Ensure you have sufficient RAM (recommended: 16GB+).
- Check that all dataset files are properly extracted.
- Verify the dataset version matches what's expected by MMDetection3D.

**Issue**: Path errors during evaluation
- Double-check all paths in `config/dataset/nuscenes.yaml`.
- Ensure the info files (`.pkl`) were generated successfully in Step 3.
