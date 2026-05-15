# Installation Guide

This guide provides step-by-step instructions for installing DejaVu and all its dependencies. The installation follows [MMDetection3D's best practices](https://mmdetection3d.readthedocs.io/en/latest/get_started.html) and uses the OpenMMLab environment as the foundation.

## Prerequisites

- **NVIDIA GPU** with CUDA support (CUDA 11.6 or 11.7 recommended)
- **Python 3.8** (via conda/miniconda)
- **Linux** (tested on Ubuntu 20.04/22.04; Ubuntu 24.04 should work with some considerations - see [README.md FAQ](../README.md#ubuntu-2404-compatibility))

**⚠️ Ubuntu 24.04 Compatibility Note**: 
The project should work on Ubuntu 24.04, but there are a few considerations:
- **CUDA Compatibility**: CUDA 11.6/11.7 may require additional setup on Ubuntu 24.04. Ensure your NVIDIA drivers are compatible (driver version ≥ 510.39.01 for CUDA 11.6).
- **Python 3.8**: Ubuntu 24.04 comes with Python 3.12 by default, but using conda/miniconda to create Python 3.8 environments (as recommended) will work fine.
- **System Libraries**: Some compiled packages (e.g., `spconv-cu113`) may need to be built from source if pre-built wheels aren't available for Ubuntu 24.04.
- **Recommended Approach**: Use conda environments to isolate dependencies, which should minimize compatibility issues.

If you encounter issues, consider using Ubuntu 20.04 or 22.04 LTS, which have been more extensively tested with the required CUDA versions.

## Step 0: Environment Setup

Follow the detailed installation guide in [`create_environment.md`](create_environment.md) for:
- NVIDIA Driver installation
- CUDA installation
- Miniconda setup

### Quick Miniconda Installation

If you need to install Miniconda:

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all
```

### CUDA Installation (if needed)

To install CUDA 11.6 only (without driver):

```bash
wget https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda_11.6.0_510.39.01_linux.run
sudo sh cuda_11.6.0_510.39.01_linux.run  # Only select CUDA 11.6, not the driver
```

Add CUDA paths:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
source ~/.bashrc
nvcc --version  # Verify installation
```

## Step 1: Create Conda Environment

You have two options for creating the conda environment:

### Option A: Quick Installation (Recommended for Exact Environment Replication)

We provide a complete conda environment file (`dependency/dejavu.yml`) that contains all packages from the working environment, including additional packages that may not be in the official MMDetection3D instructions. This ensures you get the exact same environment setup:

```bash
# Create the conda environment from the provided file
conda env create -f dependency/dejavu.yml

# Activate the environment
conda activate dejavu
```

**Note**: This environment file includes all dependencies from the working `openmmlab` environment, renamed to `dejavu` for consistency. It contains additional packages that were installed over time and may be necessary for full functionality.

**After using Option A**, you can skip to [Step 4: Install MMDetection3D](#step-4-install-mmdetection3d) since most dependencies are already installed. However, you still need to:
- Clone and install MMDetection3D from source (Step 4)
- Install OpenPCDet if needed (Step 6)
- Install MMF-JDT (Step 7)
- Download pretrained models (Step 8)

### Option B: Manual Step-by-Step Installation

Following OpenMMLab's best practices, create a conda environment with Python 3.8 and install packages step by step:

```bash
conda create --name dejavu python=3.8 -y
conda activate dejavu
```

**Note**: The environment name `dejavu` is used throughout this project. If you prefer a different name, you'll need to update the configuration files accordingly.

If you choose Option B, continue with the following steps.

## Step 2: Install PyTorch

Install PyTorch following the [official instructions](https://pytorch.org/get-started/locally/). For CUDA 11.6:

```bash
# CUDA 11.6
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116
```

**Alternative options** (if you prefer conda or different CUDA versions):

```bash
# Using conda with CUDA 11.6
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge

# Using conda with CUDA 11.7
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
```

**Note**: When installing PyTorch, you need to specify the version of CUDA. If you are not clear on which to choose:
- For Ampere-based NVIDIA GPUs (GeForce 30 series, NVIDIA A100), CUDA 11 is a must.
- For older NVIDIA GPUs, CUDA 11 is backward compatible, but CUDA 10.2 offers better compatibility and is more lightweight.

Please make sure the GPU driver satisfies the minimum version requirements. See [this table](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) for more information.

## Step 3: Install MMEngine, MMCV, and MMDetection

Following MMDetection3D's best practices, install the foundational OpenMMLab packages using MIM (MMInstall):

```bash
pip install -U openmim
mim install mmengine
mim install 'mmcv>=2.0.0rc4'
mim install 'mmdet>=3.0.0'
```

**Note**: 
- In MMCV-v2.x, `mmcv-full` is renamed to `mmcv`. If you want to install `mmcv` without CUDA ops, you can use `mim install "mmcv-lite>=2.0.0rc4"` to install the lite version.
- For DejaVu, we specifically use `mmcv==2.0.0` as specified in the README.md, so you may want to pin this version:
  ```bash
  mim install 'mmcv==2.0.0'
  ```

### Alternative: Install without MIM

If you prefer to install without MIM, you can use pip directly:

```bash
pip install mmengine
pip install "mmcv>=2.0.0rc4" -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.0/index.html
pip install "mmdet>=3.0.0"
```

**Note**: Installing MMCV with pip requires manually specifying a find-url based on PyTorch version and its CUDA version. Replace `cu116` and `torch1.12.0` with your specific versions.

## Step 4: Install MMDetection3D

MMDetection3D is used as the evaluation framework for 3D object detection. Install it from source:

```bash
# Clone MMDetection3D (used as evaluation framework)
git clone https://github.com/open-mmlab/mmdetection3d.git -b dev-1.x mmdetection3d
cd mmdetection3d
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
cd ..
```

**Note**: 
- The `-b dev-1.x` branch is used for compatibility with the project.
- Installing in editable mode (`-e`) allows you to modify MMDetection3D code if needed without reinstalling.

### Optional: Install Additional Dependencies

Some dependencies are optional. Simply running `pip install -v -e .` will only install the minimum runtime requirements. To use optional dependencies like `albumentations` and `imagecorruptions`:

```bash
pip install -r requirements/optional.txt
# or
pip install -v -e .[optional]
```

Valid keys for the extras field are: `all`, `tests`, `build`, and `optional`.

### Sparse Convolution Backends (Optional)

MMDetection3D supports multiple sparse convolution backends:

**spconv 2.0** (recommended for less GPU memory usage):
```bash
pip install cumm-cuxxx
pip install spconv-cuxxx
```
Where `xxx` is the CUDA version in the environment. For example, using CUDA 11.3:
```bash
pip install cumm-cu113 && pip install spconv-cu113
```
Supported CUDA versions include 10.2, 11.1, 11.3, and 11.4.

**Minkowski Engine**:
```bash
conda install openblas-devel -c anaconda
export CPLUS_INCLUDE_PATH=CPLUS_INCLUDE_PATH:${YOUR_CONDA_ENVS_DIR}/include
# replace ${YOUR_CONDA_ENVS_DIR} to your anaconda environment path e.g. `/home/username/anaconda3/envs/dejavu`.
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=/opt/conda/include" --install-option="--blas=openblas"
```

**Torchsparse**:
```bash
sudo apt-get install libsparsehash-dev
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
```

Or without sudo:
```bash
conda install -c bioconda sparsehash
export CPLUS_INCLUDE_PATH=CPLUS_INCLUDE_PATH:${YOUR_CONDA_ENVS_DIR}/include
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
```

## Step 5: Install Project Dependencies

Install the core Python dependencies for DejaVu:

```bash
# Make sure you're in the DejaVu root directory
pip install -r dependency/requirement.txt
```

## Step 6: Install OpenPCDet (Optional, for nuScenes evaluation)

OpenPCDet is an optional dependency used for nuScenes evaluation:

```bash
git clone https://github.com/open-mmlab/OpenPCDet.git
cd OpenPCDet
python setup.py develop
cd ..
```

## Step 7: Install MMF-JDT (Required for Multi-Object Tracking evaluation)

[MMF-JDT](https://github.com/wangxiyang2022/MMF-JDT) is a multi-modal 3D multi-object tracking framework used for evaluating tracking performance under temporal misalignment attacks.

### Option A: Using the Provided Conda Environment (Recommended)

We provide a conda environment file with all required dependencies:

```bash
# Create the conda environment from the provided file
conda env create -f dependency/mmfjdt.yml

# Activate the environment
conda activate mmfjdt
```

### Option B: Manual Installation

If you prefer to set up manually:

```bash
# Clone MMF-JDT repository
git clone https://github.com/wangxiyang2022/MMF-JDT.git
cd MMF-JDT

# Install dependencies
pip install -r requirements.txt

# Install detector dependencies (choose based on your detector)
# For OpenPCDet detector:
cd detector/pcdet
python setup.py develop
cd ../..

# For CasA detector:
# cd detector/CasA
# python setup.py develop
# cd ../..

# For TED detector:
# cd detector/TED
# python setup.py develop
# cd ../..

cd ..
```

**Important Notes**:
- Make sure MMF-JDT is cloned at the same level as the DejaVu repository (i.e., `../MMF-JDT/` relative to the DejaVu root directory), as the tracking evaluation scripts expect this path structure.
- The tracking evaluation script uses the `mmfjdt` conda environment by default. If you use a different environment, you may need to modify `src/mm_poison_and_evaluate_track_kitti_dataset.py` to point to your environment's Python executable.

## Step 8: Download Pretrained Models

Download the required pretrained model checkpoints and place them in the appropriate directories:

- **MVXNet (KITTI)**: Place in `mmdetection3d/checkpoints/`
- **BEVFusion (nuScenes)**: Place in `mmdetection3d/checkpoints/`

See the model configuration files for specific checkpoint names and download links.

## Step 9: Verify the Installation

To verify whether MMDetection3D is installed correctly, we provide some sample codes to run an inference demo.

### Step 9.1: Download Config and Checkpoint Files

```bash
cd mmdetection3d
mim download mmdet3d --config pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car --dest .
```

The downloading will take several seconds or more, depending on your network environment. When it is done, you will find two files `pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py` and `hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth` in your current folder.

### Step 9.2: Verify the Inference Demo

Run the following command:

```bash
python demo/pcd_demo.py demo/data/kitti/000008.bin pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth --show
```

You will see a visualizer interface with point cloud, where bounding boxes are plotted on cars.

**Note**:
- If you install MMDetection3D on a remote server without display device, you can leave out the `--show` argument. Demo will still save the predictions to `outputs/pred/000008.json` file.
- If you want to input a `.ply` file, you can convert it to `.bin` format using the conversion utilities provided in MMDetection3D documentation.

### Alternative: Verify with Python API

If you install MMDetection3D with MIM (not from source), you can verify using the Python API:

```python
from mmdet3d.apis import init_model, inference_detector

config_file = 'pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py'
checkpoint_file = 'hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth'
model = init_model(config_file, checkpoint_file)
inference_detector(model, 'demo/data/kitti/000008.bin')
```

You will see a list of `Det3DDataSample`, and the predictions are in the `pred_instances_3d`, indicating the detected bounding boxes, labels, and scores.

## Step 10: Configure Visualization (Required for Demos)

**⚠️ Required for Multimodal Demos and Visualizations**: Before running multimodal demos or creating visualizations, you need to configure the visualization window behavior in MMEngine's visualizer.

For automated visualization generation and batch processing, you need to modify **both** visualization files to set `wait_time` to a small positive value (e.g., `0.01`) so that visualization windows auto-close and the code continues running.

**Files to modify**:
1. **`mmengine/visualization/visualizer.py`** (typically located at `miniforge3/envs/openmmlab/lib/python3.8/site-packages/mmengine/visualization/visualizer.py`)
2. **`mmdet3d/visualization/local_visualizer.py`** (typically located at `miniforge3/envs/openmmlab/lib/python3.8/site-packages/mmdet3d/visualization/local_visualizer.py`)

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

## Troubleshooting

### Common Issues

1. **CUDA version mismatch**: Ensure your PyTorch CUDA version matches your system CUDA version. Check with:
   ```bash
   python -c "import torch; print(torch.version.cuda)"
   nvcc --version
   ```

2. **MMCV installation issues**: If you encounter issues installing MMCV, try:
   - Using MIM instead of pip: `mim install 'mmcv>=2.0.0rc4'`
   - Installing the lite version: `mim install "mmcv-lite>=2.0.0rc4"`
   - Building from source (see [MMCV installation guide](https://mmcv.readthedocs.io/en/latest/get_started/installation.html))

3. **spconv installation issues**: If pre-built wheels aren't available for your system, you may need to build from source. See the [spconv documentation](https://github.com/traveller59/spconv) for details.

4. **Import errors**: Make sure you've activated the correct conda environment and installed all dependencies in the correct order.

### Getting Help

If you encounter issues during installation:
- Check the [FAQ section in README.md](../README.md#frequently-asked-questions-faq--notes)
- Review the [IMPROVEMENTS.md](../IMPROVEMENTS.md) file for known issues
- Check the [MMDetection3D FAQ](https://mmdetection3d.readthedocs.io/en/latest/faq.html)
- Open an issue on GitHub with detailed error messages and system information

## Next Steps

After completing the installation:
1. Prepare your datasets following the [Dataset Preparation Guide](prepare_kitti.md) (for KITTI) or [nuScenes Preparation Guide](prepare_nuscenes.md)
2. Review the [Usage section in README.md](../README.md#-usage) to start running attacks
3. Configure your attack parameters in `config/dataset/kitti.yaml` or `config/dataset/nuscenes.yaml`

## References

- [MMDetection3D Installation Guide](https://mmdetection3d.readthedocs.io/en/latest/get_started.html)
- [MMEngine Documentation](https://mmengine.readthedocs.io/)
- [MMCV Documentation](https://mmcv.readthedocs.io/)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)

