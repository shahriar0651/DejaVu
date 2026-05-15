# DejaVu: Temporal Misalignment Attack against Multimodal Perception in Autonomous Driving

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

# DejaVu
Official implementation of *“Temporal Misalignment Attacks against Multimodal Perception in Autonomous Driving”* (IEEE SaTML 2026).

**DejaVu** is a security research project that demonstrates temporal misalignment attacks against multimodal 3D object detection and multi-object tracking (MOT) systems in autonomous driving. This attack exploits the critical reliance on precise time synchronization between different sensor modalities (e.g., camera and LiDAR) by introducing deliberate temporal delays, causing significant degradation in perception performance.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Attack Mechanism](#attack-mechanism)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

## 🎯 Overview

DejaVu introduces a novel attack vector that targets the temporal synchronization mechanism in multimodal autonomous driving perception systems. The attack demonstrates that:

- **3D Object Detection**: A single-frame LiDAR delay can reduce vehicle detection mAP by up to 88.5%
- **Multi-Object Tracking**: A three-frame camera delay can cause a 73% decrease in MOTA (Multiple Object Tracking Accuracy)

The project provides a comprehensive framework for:
1. **Dataset Creation & Manipulation**: Converting tracking datasets to detection format and applying temporal misalignment attacks
2. **Attack Evaluation**: Systematic evaluation across multiple models and datasets
3. **Visualization**: Tools for analyzing and visualizing attack impacts

## ✨ Key Features

- **Multi-Dataset Support**: 
  - KITTI Tracking → Detection conversion
  - nuScenes dataset support
- **Multiple Attack Modes**:
  - Constant delay attacks
  - Random delay attacks
- **Comprehensive Evaluation**:
  - 3D Object Detection (MVXNet, BEVFusion)
  - Multi-Object Tracking (MOT) using MMF-JDT
- **Flexible Configuration**: Hydra-based configuration system for easy experimentation

## 🔧 Attack Mechanism

The DejaVu attack works by introducing temporal misalignment between sensor modalities:

1. **Temporal Delay Injection**: Delays are introduced by shifting frame indices in the dataset, causing camera and LiDAR data from different timestamps to be paired together
2. **Modality-Specific Impact**: 
   - LiDAR delays primarily affect 3D object detection
   - Camera delays primarily affect multi-object tracking
3. **Attack Modes**:
   - **Constant Delay**: Fixed frame offset for all samples
   - **Random Delay**: Variable delay within a specified range

The attack can be implemented through various threat vectors:
- **C1**: Compromising time synchronization protocols (e.g., gPTP)
- **C2**: Manipulating timestamps on compromised sensor ECUs
- **C3**: ROS2 node impersonation and message injection

## 🚀 Installation

For detailed installation instructions, please refer to the comprehensive [Installation Guide](docs/Installation.md), which combines instructions from [MMDetection3D's official documentation](https://mmdetection3d.readthedocs.io/en/latest/get_started.html) with project-specific requirements.

### Quick Start

**Fastest way to get started** (recommended):

```bash
conda env create -f dependency/dejavu.yml
conda activate dejavu
```

Then proceed to install MMDetection3D, OpenPCDet (optional), and MMF-JDT as described in the [Installation Guide](docs/Installation.md).

### Quick Overview

**Prerequisites**:
- **NVIDIA GPU** with CUDA support (CUDA 11.6 or 11.7 recommended)
- **Python 3.8** (via conda/miniconda)
- **Linux** (tested on Ubuntu 20.04/22.04; Ubuntu 24.04 should work with some considerations - see [FAQ](#-frequently-asked-questions-faq--notes))

**Installation Options** (see [Installation.md](docs/Installation.md) for details):

**Quick Installation** (Recommended):
- Use the provided conda environment file: `conda env create -f dependency/dejavu.yml`
- This creates the `dejavu` environment with all dependencies from the working environment
- Then proceed to install MMDetection3D, OpenPCDet (optional), and MMF-JDT

**Manual Installation** (Step-by-step):
1. **Environment Setup**: NVIDIA Driver, CUDA, and Miniconda installation
2. **Create Conda Environment**: `conda create --name dejavu python=3.8 -y`
3. **Install PyTorch**: With CUDA 11.6 support
4. **Install MMEngine, MMCV, MMDetection**: Using MIM (MMInstall)
5. **Install MMDetection3D**: From source (dev-1.x branch)
6. **Install Project Dependencies**: Core DejaVu requirements
7. **Install OpenPCDet** (Optional): For nuScenes evaluation
8. **Install MMF-JDT**: For multi-object tracking evaluation
9. **Download Pretrained Models**: MVXNet and BEVFusion checkpoints
10. **Configure Visualization**: Modify visualizer files for batch processing

**⚠️ Important**: Before running multimodal demos or visualizations, you must configure the visualization window behavior as described in the [Visualization Configuration](#visualization-configuration) section below.

## 📦 Dataset Preparation


For detailed instructions, see [`docs/prepare_kitti.md`](docs/prepare_kitti.md).

### nuScenes Dataset
For detailed instructions, see [`docs/prepare_nuscenes.md`](docs/prepare_nuscenes.md).

## 💻 Usage

### 3D Object Detection Attack (KITTI with MVXNet)

1. **Convert KITTI Tracking Dataset** (if not done already):
   ```bash
   cd src
   python 0_convert_kitti_tracking.py
   ```

2. **Run Attack Evaluation**:
   
   The pipeline runs 4 steps: (1) poison dataset, (2) finalize dataset, (3) evaluate models, (4) visualize results. You can control which steps run using the `evaluate` and `visualize` flags:
   
   **Run all steps (default: steps 1, 2, 3)**:
   ```bash 
   python mm_poison_and_evaluate_detect_kitti_dataset.py \
      -m dataset=kitti \
      dataset.delays.image=0,1,2,3,4,5 \
      dataset.delays.lidar=0,1,2,3,4,5 
    ```

   **Run steps 1, 2, 3, 4 (evaluation + visualization)**:
   ```bash 
   python mm_poison_and_evaluate_detect_kitti_dataset.py \
      -m dataset=kitti \
      dataset.delays.lidar=0,1,2,3,4,5 \
      dataset.delays.image=0,1,2,3,4,5 \
      dataset.evaluate=False \
      dataset.visualize=True \
      dataset.wait_time=0.01
    ```
   
   **⚠️ Display Requirement**: When `visualize=True`, the machine must have a display/head available (not a remote headless server) as visualization requires a GUI environment to create visual outputs.


3. **Visualize Results**:
   ```bash
   jupyter notebook ../notebooks/visualize_detection_results_kitti_mm.ipynb
   ```
   
   **Note**: Before running visualizations, make sure to configure the visualization window behavior as described in the [Visualization Configuration](#visualization-configuration) section.

### 3D Object Detection Attack (nuScenes with BEVFusion)

**Using MMDetection3D**:
```bash
cd src
nohup python mm_poison_and_evaluate_detect_nusc_dataset.py \
    -m dataset=nuscenes \
    dataset.delays.image=3,4,5 \
    dataset.delays.lidar=1,2,3,4,5 &
```

**Using OpenPCDet**:
```bash
conda activate mmfjdt  # or your OpenPCDet environment
cd src
python pc_poison_and_evaluate_nuscenes_dataset.py \
    -m dataset=nuscenes \
    dataset.delays.image=0,1,2,3,4,5 \
    dataset.delays.lidar=0,1,2,3,4,5
```

### Multi-Object Tracking Attack (KITTI)

The tracking attack evaluation uses [MMF-JDT](https://github.com/wangxiyang2022/MMF-JDT) for multi-object tracking. 

**Prerequisites**:
1. Installed MMF-JDT (see [Installation Step 6](#step-6-install-mmf-jdt-required-for-multi-object-tracking-evaluation))
2. Created and activated the `mmfjdt` conda environment (from `dependency/mmfjdt.yml`)
3. Configured the dataset path in `MMF-JDT/config/kitti.yaml`

**Run the attack**:
```bash
# Make sure the mmfjdt environment is activated
conda activate mmfjdt

cd src
python mm_poison_and_evaluate_track_kitti_dataset.py \
    -m dataset.delays.image=0,1,2,3,4,5 \
    dataset.delays.lidar=0,1,2,3,4,5 \
    dataset.att_mode=constant
```python mm_poison_and_evaluate_detect_kitti_dataset.py \
       -m dataset=kitti \

**Note**: The script automatically uses the `mmfjdt` environment's Python executable. If you need to use a different environment, modify the `python_mot_env` variable in `src/mm_poison_and_evaluate_track_kitti_dataset.py`.

**What the script does**:
1. Applies temporal delays to the KITTI tracking dataset
2. Runs MMF-JDT's `kitti_main.py` for tracking evaluation
3. Saves results to `MMF-JDT/output/` (renamed with attack parameters)

**Results**: Tracking evaluation results are saved in `MMF-JDT/output/` (or `MMF-JDT/new_output_<attack_params>/` after each run) and evaluated using HOTA metrics.

### Visualization Configuration

**⚠️ Required for Multimodal Demos and Visualizations**: Before running multimodal demos (e.g., `multi_modal_cont_demo.py`) or creating visualizations, you need to configure the visualization window behavior in MMEngine's visualizer.

The MMEngine visualizer controls whether visualization windows auto-close or stay open. This configuration is essential for:
- Running multimodal detection demos
- Creating visualization outputs
- Interactive result inspection

**To configure visualization window behavior**:

For automated visualization generation and batch processing, you need to modify **both** visualization files to set `wait_time` to a small positive value (e.g., `0.01`) so that visualization windows auto-close and the code continues running.

**Files to modify**:
1. **`mmengine/visualization/visualizer.py`** (typically located at `miniconda3/envs/dejavu/lib/python3.8/site-packages/mmengine/visualization/visualizer.py`)
2. **`mmdet3d/visualization/local_visualizer.py`** (typically located at `miniconda3/envs/dejavu/lib/python3.8/site-packages/mmdet3d/visualization/local_visualizer.py`)


**Reference implementations**: Modified versions of both files are provided in the `misc/` folder as reference:
- `misc/visualizer.py` - Modified MMEngine visualizer
- `misc/local_visualizer.py` - Modified MMDetection3D local visualizer

**Modification instructions**:

1. **In `mmengine/visualization/visualizer.py`** (around line 252-254):
```python
# TODO : Hasan: 
wait_time = 0.01  # Set to 0.01 to auto-close window and continue execution
#-----------------
```

2. **In `mmdet3d/visualization/local_visualizer.py`** (around line 819):
```python
wait_time = 0.01  # Set to 0.01 to auto-close window and continue execution
# TODO: Hasan To destroy figures after 0.01 second (-1 to no destroy)
```

**Configuration options**:
- **`wait_time = 0.01`** (recommended for batch processing): Auto-closes the visualization window after 0.01 seconds, allowing the code to continue running automatically. This is essential for generating visualizations in batch mode.
- **`wait_time = -1`**: Keeps the visualization window open indefinitely (useful only for interactive inspection of individual samples, but will block batch processing)

**Note**: This modification is required to run multimodal demos and create visualizations. Adjust this value based on whether you're running interactive demos or batch evaluations.

### Configuration

The attack parameters can be configured through Hydra configuration files:

- **Main config**: `config/config.yaml`
- **KITTI config**: `config/dataset/kitti.yaml`
- **nuScenes config**: `config/dataset/nuscenes.yaml`

Key parameters:
- `dataset.delays.image`: Camera delay in frames (comma-separated list)
- `dataset.delays.lidar`: LiDAR delay in frames (comma-separated list)
- `dataset.att_mode`: Attack mode (`constant` or `random`)
- `dataset.maxTrip`: Maximum number of sequences to process (-1 for all)
- `dataset.evaluate`: Enable/disable evaluation step (default: `True`). Set to `False` to skip model evaluation (step 3)
- `dataset.visualize`: Enable/disable visualization step (default: `False`). Set to `True` to generate visualizations (step 4)
  - **⚠️ Important**: When `visualize=True`, the machine must be running with a display/head (not a remote headless server) as visualization requires a GUI environment to create visual outputs

## 📁 Project Structure

```
DejaVu/
├── config/                 # Configuration files (Hydra)
│   ├── config.yaml         # Main configuration
│   └── dataset/            # Dataset-specific configs
│       ├── kitti.yaml
│       └── nuscenes.yaml
├── src/                    # Main source code
│   ├── 0_convert_kitti_tracking.py      # Convert tracking to detection format
│   ├── 1_poison_kitti_tracking.py        # Apply temporal delays (detection)
│   ├── 2_finalize_kitti_tracking.py     # Finalize dataset preparation
│   ├── 3_evaluate_kitti_tracking.py      # Evaluate detection models
│   ├── mm_poison_and_evaluate_detect_kitti_dataset.py    # Detection attack pipeline (KITTI)
│   ├── mm_poison_and_evaluate_detect_nusc_dataset.py     # Detection attack pipeline (nuScenes)
│   ├── mm_poison_and_evaluate_track_kitti_dataset.py     # MOT attack pipeline
│   ├── t_poison_kitti_multi_orgtracking.py               # MOT poisoning with random delays
│   ├── utils_helper_fnc.py              # Utility functions
│   └── vis/                              # Visualization tools
├── docs/                   # Documentation
│   ├── create_environment.md
│   ├── prepare_kitti.md
│   ├── prepare_nuscenes.md
│   └── understand_results.md
├── notebooks/              # Jupyter notebooks for visualization
├── artifacts/              # Results and plots
│   ├── results/            # Evaluation results (CSV)
│   └── plots/              # Visualization plots
├── mmdetection3d/          # MMDetection3D framework (external dependency)
└── mmdetection3d_209/      # Alternative MMDetection3D version (external dependency)
```

## 📊 Results

Evaluation results are stored in:
- **KITTI Detection**: `artifacts/results/kitti_tracking_results_total.csv`
- **nuScenes Detection**: `artifacts/results/results_total_nuscenes.csv`
- **Visualization plots**: `artifacts/plots/`

### Key Findings

- **3D Detection**: LiDAR delays cause severe performance degradation (up to 88.5% mAP drop)
- **MOT**: Camera delays significantly impact tracking accuracy (up to 73% MOTA decrease)
- **Modality Sensitivity**: Different tasks show different sensitivity to modality-specific delays

## 📝 Citation

If you use DejaVu in your research, please cite:

```bibtex
@article{dejavu2026,
  title={Temporal Misalignment Attacks against Multimodal Perception in Autonomous Driving},
  author={Your Name and Collaborators},
  journal={Conference/Journal Name},
  year={2026}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) for the 3D detection framework
- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) for additional evaluation support
- [MMF-JDT](https://github.com/wangxiyang2022/MMF-JDT) for the multi-modal 3D multi-object tracking framework
- KITTI and nuScenes datasets for providing benchmark datasets

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

---

**Note**: This is a research project for security analysis. Use responsibly and only on systems you own or have explicit permission to test.

## ❓ Frequently Asked Questions (FAQ) & Notes

### Ubuntu 24.04 Compatibility

**Q: Can I run DejaVu on Ubuntu 24.04?**

**A:** Yes, but with some considerations:

- **✅ What works well:**
  - Python 3.8 via conda/miniconda (Ubuntu 24.04 comes with Python 3.12, but conda environments work fine)
  - Most Python packages when installed in isolated conda environments
  - Core project functionality

- **⚠️ Potential challenges:**
  - **CUDA 11.6/11.7 compatibility**: Ubuntu 24.04 may not have official support for these older CUDA versions. You may need to:
    - Ensure NVIDIA drivers are compatible (driver version ≥ 510.39.01 for CUDA 11.6)
    - Install CUDA manually rather than through package manager
    - Consider using CUDA 12.x with updated PyTorch (requires package updates)
  
  - **Compiled packages**: Some CUDA-dependent packages like `spconv-cu113` may need to be built from source if pre-built wheels aren't available for Ubuntu 24.04
  
  - **System libraries**: Newer system library versions (glibc, etc.) might cause conflicts with older compiled packages, though conda environments help mitigate this

- **💡 Recommendations:**
  - Use conda environments (already recommended) to isolate dependencies
  - Test CUDA installation first: `nvidia-smi` and `nvcc --version`
  - If you encounter issues, consider:
    - Ubuntu 22.04 LTS (better tested with CUDA 11.6/11.7)
    - Using Docker containers with a tested Ubuntu version
    - Upgrading to CUDA 12.x (requires updating PyTorch and related packages)

**Bottom line**: The project should work on Ubuntu 24.04, but you may need to spend extra time on CUDA setup. For the smoothest experience, Ubuntu 20.04 or 22.04 LTS are recommended.

### Other Common Questions

**Q: Why do I need to modify the visualizer.py file?**

**A:** The MMEngine visualizer's default window behavior may not suit all use cases. Setting `wait_time = -1` keeps windows open for interactive inspection, while positive values auto-close them for batch processing. This is required for running multimodal demos and creating visualizations.

**Q: Can I use a different Python version?**

**A:** The project is designed for Python 3.8. While Python 3.9 might work, we recommend sticking with Python 3.8 for compatibility with all dependencies, especially older packages like `numba==0.48.0` and `numpy==1.23.3`.

**Q: What if I don't have an NVIDIA GPU?**

**A:** The project requires CUDA for GPU acceleration. While some components might run on CPU, performance will be significantly degraded and some features may not work. An NVIDIA GPU with CUDA support is essential for this project.

**Q: Can I use CUDA 12.x instead of 11.6/11.7?**

**A:** Yes, but you'll need to:
- Update PyTorch installation to use CUDA 12.x
- Update CUDA-dependent packages (e.g., `spconv` versions)
- Verify compatibility with MMDetection3D and MMF-JDT
- Test thoroughly as this combination hasn't been extensively tested

**Q: Where can I get help if I encounter issues?**

**A:** 
- Check the [IMPROVEMENTS.md](IMPROVEMENTS.md) file for known issues and potential solutions
- Review the documentation in the `docs/` directory
- Open an issue on GitHub with detailed error messages and system information
