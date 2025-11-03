# Installation Guide

This guide provides detailed instructions for installing the PPO-GNN framework for fuel delivery optimization.

---

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Installation](#quick-installation)
3. [Detailed Installation](#detailed-installation)
   - [Linux / macOS](#linux--macos)
   - [Windows](#windows)
   - [GPU Support](#gpu-support)
4. [Installing Gurobi (Optional)](#installing-gurobi-optional)
5. [Verification](#verification)
6. [Troubleshooting](#troubleshooting)
7. [Docker Installation (Alternative)](#docker-installation-alternative)

---

## 🖥️ System Requirements

### Minimum Requirements

- **OS:** Linux (Ubuntu 18.04+), macOS (10.15+), Windows 10/11
- **Python:** 3.8 or higher
- **RAM:** 8 GB minimum, 16 GB recommended
- **Disk Space:** 5 GB for dependencies + datasets
- **CPU:** Multi-core processor (4+ cores recommended)

### Recommended Requirements (with GPU)

- **GPU:** NVIDIA GPU with CUDA support (compute capability 6.0+)
- **CUDA:** 11.6 or higher
- **cuDNN:** 8.0 or higher
- **RAM:** 16 GB minimum, 32 GB recommended
- **VRAM:** 8 GB minimum (NVIDIA RTX 3060 or equivalent)

### Software Dependencies

- **Python packages:** See `requirements.txt`
- **Gurobi:** 10.0+ (optional, requires license for exact solver comparison)
- **Git:** For cloning the repository

---

## ⚡ Quick Installation

### Option 1: Pip Installation (CPU-only)
```bash
# Clone repository
git clone https://github.com/YourUsername/PPO-GNN-Fuel-Delivery.git
cd PPO-GNN-Fuel-Delivery

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch_geometric; print(f'PyTorch Geometric: {torch_geometric.__version__}')"
```

### Option 2: Conda Installation (Recommended)
```bash
# Clone repository
git clone https://github.com/YourUsername/PPO-GNN-Fuel-Delivery.git
cd PPO-GNN-Fuel-Delivery

# Create conda environment
conda env create -f environment.yml
conda activate ppo_gnn

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

---

## 📦 Detailed Installation

### Linux / macOS

#### Step 1: Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y python3.8 python3-pip python3-venv git
sudo apt install -y build-essential libssl-dev libffi-dev python3-dev
```

**macOS (with Homebrew):**
```bash
brew update
brew install python@3.8 git
```

#### Step 2: Clone Repository
```bash
git clone https://github.com/YourUsername/PPO-GNN-Fuel-Delivery.git
cd PPO-GNN-Fuel-Delivery
```

#### Step 3: Create Virtual Environment

**Using venv:**
```bash
python3.8 -m venv venv
source venv/bin/activate
```

**Using conda:**
```bash
conda create -n ppo_gnn python=3.8
conda activate ppo_gnn
```

#### Step 4: Install PyTorch

**CPU-only:**
```bash
pip install torch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 \
    --index-url https://download.pytorch.org/whl/cpu
```

**With CUDA 11.6:**
```bash
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 \
    --extra-index-url https://download.pytorch.org/whl/cu116
```

**With CUDA 11.3:**
```bash
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 \
    --extra-index-url https://download.pytorch.org/whl/cu113
```

#### Step 5: Install PyTorch Geometric
```bash
# Install PyTorch Geometric dependencies
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-1.12.0+cu116.html

# Install PyTorch Geometric
pip install torch-geometric==2.1.0
```

> **Note:** Replace `cu116` with your CUDA version (e.g., `cu113`, `cpu`).

#### Step 6: Install Remaining Dependencies
```bash
pip install -r requirements.txt
```

#### Step 7: Install Package in Development Mode
```bash
pip install -e .
```

---

### Windows

#### Step 1: Install Prerequisites

1. **Install Python 3.8+:** Download from [python.org](https://www.python.org/downloads/)
   - ✅ Check "Add Python to PATH" during installation
   
2. **Install Git:** Download from [git-scm.com](https://git-scm.com/downloads)

3. **Install Microsoft Visual C++ 14.0+:** Download from [Visual Studio](https://visualstudio.microsoft.com/downloads/)
   - Select "Desktop development with C++"

#### Step 2: Clone Repository
```powershell
git clone https://github.com/YourUsername/PPO-GNN-Fuel-Delivery.git
cd PPO-GNN-Fuel-Delivery
```

#### Step 3: Create Virtual Environment
```powershell
python -m venv venv
venv\Scripts\activate
```

#### Step 4: Install PyTorch

**CPU-only:**
```powershell
pip install torch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 `
    --index-url https://download.pytorch.org/whl/cpu
```

**With CUDA 11.6:**
```powershell
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 `
    --extra-index-url https://download.pytorch.org/whl/cu116
```

#### Step 5: Install PyTorch Geometric
```powershell
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv `
    -f https://data.pyg.org/whl/torch-1.12.0+cu116.html

pip install torch-geometric==2.1.0
```

#### Step 6: Install Remaining Dependencies
```powershell
pip install -r requirements.txt
```

#### Step 7: Install Package
```powershell
pip install -e .
```

---

### GPU Support

#### Check CUDA Availability
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA version
nvcc --version

# Verify PyTorch GPU support
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA Version: {torch.version.cuda}')"
python -c "import torch; print(f'Device Count: {torch.cuda.device_count()}')"
python -c "import torch; print(f'Device Name: {torch.cuda.get_device_name(0)}')"
```

#### Install CUDA Toolkit (if needed)

**Ubuntu:**
```bash
# CUDA 11.6
wget https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda_11.6.0_510.39.01_linux.run
sudo sh cuda_11.6.0_510.39.01_linux.run

# Add to ~/.bashrc
export PATH=/usr/local/cuda-11.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH
```

**Windows:**
Download CUDA Toolkit from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)

#### Install cuDNN (Optional, for optimized performance)

1. Download cuDNN from [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)
2. Extract and copy files to CUDA installation directory

---

## 🔧 Installing Gurobi (Optional)

Gurobi is required **only** for exact solver comparison experiments. The PPO-GNN framework works without it.

### Step 1: Get Gurobi License

**Academic License (Free):**
1. Register at [Gurobi Academia](https://www.gurobi.com/academia/academic-program-and-licenses/)
2. Request free academic license
3. Download license file

**Commercial License:**
Visit [Gurobi Downloads](https://www.gurobi.com/downloads/)

### Step 2: Install Gurobi
```bash
pip install gurobipy
```

### Step 3: Activate License
```bash
# Navigate to license directory
cd ~/Downloads

# Activate license (replace with your key)
grbgetkey YOUR-LICENSE-KEY

# Verify installation
python -c "import gurobipy; print(f'Gurobi Version: {gurobipy.gurobi.version()}')"
```

### Step 4: Set Environment Variables

Add to `~/.bashrc` (Linux/macOS) or System Environment Variables (Windows):
```bash
export GUROBI_HOME="/opt/gurobi1000/linux64"
export PATH="${PATH}:${GUROBI_HOME}/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
```

---

## ✅ Verification

### Test Installation
```bash
# Run verification script
python scripts/verify_installation.py
```

**Expected output:**
```
✓ Python version: 3.8.10
✓ PyTorch version: 1.12.0
✓ PyTorch Geometric version: 2.1.0
✓ CUDA available: True
✓ CUDA version: 11.6
✓ GPU device: NVIDIA GeForce RTX 3090
✓ NumPy version: 1.23.0
✓ Pandas version: 1.4.3
✓ Matplotlib version: 3.5.2
✓ NetworkX version: 2.8.4
✓ TensorBoard version: 2.9.1
✓ Gurobi version: 10.0.0 (optional)

All dependencies installed successfully!
```

### Manual Verification
```python
# test_imports.py
import torch
import torch_geometric
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import yaml
from tqdm import tqdm

print("✓ All core packages imported successfully!")

# Test CUDA
if torch.cuda.is_available():
    print(f"✓ CUDA is available: {torch.cuda.get_device_name(0)}")
    # Test simple tensor operation
    x = torch.randn(100, 100).cuda()
    y = torch.randn(100, 100).cuda()
    z = torch.mm(x, y)
    print("✓ GPU computation test passed!")
else:
    print("⚠ CUDA not available, using CPU")

# Test PyTorch Geometric
from torch_geometric.data import Data
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
x = torch.randn(3, 16)
data = Data(x=x, edge_index=edge_index)
print(f"✓ PyTorch Geometric test passed: {data}")

# Optional: Test Gurobi
try:
    import gurobipy
    print(f"✓ Gurobi available: {gurobipy.gurobi.version()}")
except ImportError:
    print("⚠ Gurobi not installed (optional)")
```

Run test:
```bash
python test_imports.py
```

---

## 🔧 Troubleshooting

### Common Issues

#### Issue 1: PyTorch Geometric Installation Fails

**Error:**
```
ERROR: Could not find a version that satisfies the requirement torch-scatter
```

**Solution:**
Ensure PyTorch is installed first, then install PyG with correct CUDA version:
```bash
pip install torch==1.12.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
```

---

#### Issue 2: CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size in config:
```yaml
   training:
     batch_size: 128  # Instead of 256
```

2. Use gradient accumulation:
```yaml
   training:
     gradient_accumulation_steps: 2
```

3. Enable mixed precision training:
```yaml
   training:
     mixed_precision: true
```

---

#### Issue 3: Import Error for torch_geometric

**Error:**
```
ModuleNotFoundError: No module named 'torch_geometric'
```

**Solution:**
```bash
pip uninstall torch-geometric
pip install torch-geometric==2.1.0
```

---

#### Issue 4: Gurobi License Error

**Error:**
```
GurobiError: No Gurobi license found
```

**Solution:**
1. Verify license file location:
```bash
   ls ~/.gurobi/
```

2. Check environment variable:
```bash
   echo $GRB_LICENSE_FILE
```

3. Re-activate license:
```bash
   grbgetkey YOUR-LICENSE-KEY
```

---

#### Issue 5: Version Conflicts

**Error:**
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed
```

**Solution:**
Use clean environment:
```bash
# Remove old environment
conda env remove -n ppo_gnn

# Create fresh environment
conda create -n ppo_gnn python=3.8
conda activate ppo_gnn

# Install from scratch
pip install -r requirements.txt
```

---

#### Issue 6: Windows Long Path Error

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory
```

**Solution:**
Enable long paths in Windows:
1. Run as Administrator:
```powershell
   New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
     -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```
2. Restart computer

---

### Platform-Specific Issues

#### macOS M1/M2 (Apple Silicon)
```bash
# Use ARM-native PyTorch
conda install pytorch torchvision torchaudio -c pytorch

# Install PyG from source
pip install torch-geometric
```

#### WSL2 (Windows Subsystem for Linux)
```bash
# Enable GPU support in WSL2
# Follow: https://docs.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute

# Verify CUDA
nvidia-smi
```

---

## 🐳 Docker Installation (Alternative)

### Option 1: Use Pre-built Image
```bash
# Pull image
docker pull yourrepo/ppo-gnn-fuel-delivery:latest

# Run container
docker run --gpus all -it -v $(pwd):/workspace yourrepo/ppo-gnn-fuel-delivery:latest
```

### Option 2: Build from Dockerfile
```bash
# Build image
docker build -t ppo-gnn-fuel-delivery .

# Run container with GPU support
docker run --gpus all -it \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/results:/workspace/results \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  ppo-gnn-fuel-delivery
```

### Dockerfile
```dockerfile
# Base image with CUDA support
FROM nvidia/cuda:11.6.0-cudnn8-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 python3-pip git wget \
    build-essential libssl-dev libffi-dev python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace/

# Install Python dependencies
RUN pip3 install --upgrade pip && \
    pip3 install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 \
      --extra-index-url https://download.pytorch.org/whl/cu116 && \
    pip3 install torch-scatter torch-sparse torch-cluster torch-spline-conv \
      -f https://data.pyg.org/whl/torch-1.12.0+cu116.html && \
    pip3 install -r requirements.txt && \
    pip3 install -e .

# Expose ports for TensorBoard
EXPOSE 6006

# Default command
CMD ["/bin/bash"]
```

---

## 📚 Next Steps

After successful installation:

1. **Generate Datasets:**
```bash
   python data/generation_scripts/generate_networks.py --num_nodes 100
```

2. **Run Demo Training:**
```bash
   python src/training/train_ppo_gnn.py --config configs/ppo_gnn_config.yaml
```

3. **Explore Tutorials:**
```bash
   jupyter notebook notebooks/01_demo_training.ipynb
```

4. **Read Documentation:**
   - [REPRODUCTION.md](REPRODUCTION.md) - Reproduce paper results
   - [API.md](API.md) - Code documentation
   - [FAQ.md](FAQ.md) - Frequently asked questions

---

## 💬 Support

If you encounter issues not covered here:

1. **Check existing issues:** [GitHub Issues](https://github.com/YourUsername/PPO-GNN-Fuel-Delivery/issues)
2. **Open new issue:** Provide error message, OS, Python version, and steps to reproduce
3. **Ask in discussions:** [GitHub Discussions](https://github.com/YourUsername/PPO-GNN-Fuel-Delivery/discussions)
4. **Contact authors:** See README.md for contact information

---

## 🔄 Updating Installation
```bash
# Update repository
git pull origin main

# Update dependencies
pip install --upgrade -r requirements.txt

# Reinstall package
pip install -e .
```

---

## 🗑️ Uninstallation
```bash
# Using pip
pip uninstall ppo-gnn-fuel-delivery

# Remove conda environment
conda env remove -n ppo_gnn

# Remove repository
rm -rf PPO-GNN-Fuel-Delivery
```

---

**Installation complete! Ready to optimize fuel delivery routes! 🚀**
