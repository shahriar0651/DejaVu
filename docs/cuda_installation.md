Implementation Note: Installing CUDA Toolkit 11.7 on Ubuntu 20.04
1. Verify Existing NVIDIA Setup

Before starting, check that the NVIDIA GPU and driver are recognized:

nvidia-smi
nvcc --version

2. Prepare the System for CUDA Installation

Navigate to your home directory and clean the terminal:

cd ~
clear


Install prerequisites:

sudo apt install -y ca-certificates curl gnupg
sudo mkdir -p /usr/share/keyrings

3. Add NVIDIA CUDA Repository

Download the repository pin:

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600


Add NVIDIA public key:

curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-cuda-keyring.gpg


Add the repository to apt sources:

echo "deb [signed-by=/usr/share/keyrings/nvidia-cuda-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" | sudo tee /etc/apt/sources.list.d/nvidia-cuda.list


Update package lists:

sudo apt update

4. Install CUDA Toolkit 11.7
sudo apt install -y cuda-toolkit-11-7

5. Verify Installation

Check CUDA compiler and driver:

nvcc --version
nvidia-smi

6. Configure Environment Variables

Add CUDA binaries and libraries to the environment:

export PATH=/usr/local/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH


Make these changes permanent by appending them to .bashrc:

echo 'export PATH=/usr/local/cuda-11.7/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

7. Reboot

After installation and environment setup:

sudo reboot


Verify after reboot:

nvidia-smi
nvcc --version


✅ Notes:

Remove old or temporary CUDA repository files to prevent conflicts.

Ensure your GPU driver is compatible with CUDA 11.7.

Use /usr/local/cuda-11.7/bin explicitly if multiple CUDA versions exist.

If you want, I can also make an even shorter “one-shot script” version that installs everything automatically without needing to run 15+ separate commands manually.