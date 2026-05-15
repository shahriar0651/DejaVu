# Create the environment

### Install Miniconda
- Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all
```


### Create a conda environment and activate it.

```shell
conda create --name dejavu python=3.8 -y
conda activate dejavu
```

- Note: You will also need to install `unitr` for OpenPCDet.


### Install PyTorch 
Follow [official instructions](https://pytorch.org/get-started/locally/), e.g. on GPU platforms:

```shell
# CUDA 11.6
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116
# conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
# conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
# conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
```



Tips:

to install 11.7:
```bash
wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run 
sudo sh cuda_11.7.0_515.43.04_linux.run 
```

Tips: To install CUDA 11.6 only
```bash
wget https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda_11.6.0_510.39.01_linux.run
sudo sh cuda_11.6.0_510.39.01_linux.run #Only select the CUDA 11.6 not the driver
```

Add paths to the source:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
source ~/.bashrc
nvcc --version
```