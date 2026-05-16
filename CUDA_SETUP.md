# CUDA Setup Guide for Aksharam Project

This guide enables GPU acceleration for the Malayalam translation training pipeline.

## Prerequisites

- NVIDIA GPU (GTX 1050 or better recommended)
- 8GB+ GPU VRAM (16GB+ recommended for optimal performance)

## Installation Steps

### 1. Install NVIDIA CUDA Toolkit and cuDNN

**For Windows:**

1. Download CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
   - Select Windows platform, your OS version, and runtime installer
   - Install to default location (usually `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`)

2. Download cuDNN: https://developer.nvidia.com/cudnn
   - Extract to CUDA installation directory

3. Verify installation:
   ```powershell
   nvidia-smi
   nvcc --version
   ```

### 2. Update Python Environment

**Option A: Fresh Virtual Environment (Recommended)**

```powershell
# Create fresh venv
python -m venv venv_cuda
.\venv_cuda\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt
```

**Option B: Update Existing Environment**

```powershell
# Activate your current venv
.\venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Reinstall pytorch with CUDA
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# If needed, update other deps
pip install -r requirements.txt
```

### 3. Verify CUDA is Working

```powershell
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}'); print(f'CUDA Version: {torch.version.cuda}')"
```

Expected output:
```
CUDA Available: True
GPU: NVIDIA GeForce RTX 3060
CUDA Version: 12.1
```

## Training with CUDA

The training scripts now automatically detect and use GPU:

```powershell
# Collect and align data (uses GPU for embeddings)
python 01_collect_and_align.py

# Train model with LoRA (uses GPU for compute)
python 02_train.py
```

### GPU Optimization Tips

1. **Batch Size**: If you get Out-Of-Memory (OOM) errors, reduce `per_device_train_batch_size` in training scripts
   - Start with 4 or 2 if 8 causes issues

2. **Memory Monitoring**: Check GPU memory usage during training
   ```powershell
   # In separate terminal
   nvidia-smi -l 1  # Updates every 1 second
   ```

3. **Mixed Precision Training**: Already enabled when CUDA is available (`fp16=True`)
   - Saves 50% GPU memory while maintaining performance

4. **Gradient Accumulation**: For larger effective batch sizes without OOM
   - Add `gradient_accumulation_steps=2` to training args

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `CUDA out of memory` | Reduce batch size: `per_device_train_batch_size=4` |
| `CUDA not available` | Reinstall PyTorch with correct index: `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| `NVIDIA driver error` | Update driver: https://www.nvidia.com/Download/driverDetails.aspx |
| `nvcc not found` | Add CUDA to PATH: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin` |

## Performance Notes

- GPU training is **10-100x faster** than CPU
- mBART-50 with LoRA typically trains in **2-5 minutes per epoch** on modern GPUs
- BLEU/LaBSE evaluation also benefits from GPU acceleration

## CPU Fallback

If GPU is unavailable, training automatically falls back to CPU. To force CPU:
```python
device = "cpu"
```
