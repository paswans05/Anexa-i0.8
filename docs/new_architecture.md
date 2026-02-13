# Anexa: New Architecture (PyTorch)

This document describes the **modern, production-grade version** of Anexa. This version uses PyTorch for high-performance training and inference.

## Key Features

### 1. PyTorch Engine
We replaced the custom `Value` class with `torch.Tensor`.
-   **Autograd**: PyTorch handles backpropagation automatically and efficiently.
-   **Optimization**: Operations like Matrix Multiplication (`@`) dispatch to highly optimized C++ (BLAS) or CUDA kernels.

### 2. Transformer Implementation
We use standard PyTorch modules:
-   `nn.Embedding`: For token and position embeddings.
-   `nn.Linear`: For projections in Attention and Feed-Forward layers.
-   `nn.LayerNorm`: For normalization.
-   `F.softmax` / `F.cross_entropy`: For probability calculations and loss.

### 3. GPU Acceleration
-   **Device Agnostic**: The code automatically detects if a CUDA-enabled GPU (like NVIDIA RTX 3050) is available.
-   **Speed**: Training is ~10,000x faster than the pure Python version.

### 4. SafeTensors Persistence
We use the `safetensors` library for saving/loading the model.
-   **Safety**: Prevents arbitrary code execution vulnerabilities common with `pickle` (`torch.load`).
-   **Speed**: Uses zero-copy memory mapping for instant loading.
-   **Format**: The model is saved as `model.safetensors`.

## Usage

### Installation
```bash
# Create environment
python -m venv venv
venv\Scripts\activate

# Install dependencies (with CUDA 11.8)
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install safetensors packaging
```

### Training
```bash
python anexagpt.py
```
This will:
1.  Download "The Adventures of Sherlock Holmes".
2.  Train the model on your GPU.
3.  Save the weights to `model.safetensors`.

### Inference
After training, the script enters an **interactive loop**:
```
Prompt: Sherlock
Response: Sherlock Holmes, who was usually very late in the mornings...
```
