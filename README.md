# Anexa (formerly MicroGPT)

![Anexa Banner](https://img.shields.io/badge/Model-Anexa-blue?style=for-the-badge) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white) ![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Anexa** is a minimal, educational, yet production-grade implementation of a GPT (Generative Pre-trained Transformer) language model.

It began as a "from scratch" Pure Python project to understand the math behind AI, and has evolved into a high-performance **PyTorch** model capable of training on GPUs and generating coherent text.

## 🚀 Why Anexa?

In a world of closed-source giants like ChatGPT and Gemini, Anexa stands for:

*   **Understanding**: Built line-by-line to teach you how Transformers work.
*   **Speed**: Runs locally on your machine. No queues, no latency, no internet required.
*   **Privacy**: Your data never leaves your laptop.
*   **Control**: You own the weights. You own the code.

## 📚 Documentation

We have detailed documentation for both versions of the architecture:

*   **[Old Architecture (Pure Python)](docs/old_architecture.md)**: The original educational version. Learn how backpropagation and matrix multiplication work under the hood without any libraries.
*   **[New Architecture (PyTorch)](docs/new_architecture.md)**: The modern version of Anexa. Features **GPU acceleration**, **SafeTensors** persistence, and standard PyTorch layers.

## ⚡ Quick Start

Anexa uses PyTorch and SafeTensors for maximum performance.

### 1. Installation

Create a virtual environment and install dependencies (requires NVIDIA GPU for CUDA support):

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Install PyTorch (CUDA 11.8) and SafeTensors
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install safetensors packaging
```

### 2. Usage

Run the model script. It will automatically download "The Adventures of Sherlock Holmes", train the model for 5000 steps, and save it to `model.safetensors`.

```bash
python anexagpt.py
```

### 3. Interactive Mode

After training, the script enters an interactive mode where you can chat with the model:

```
Training finished...
Model saved to model.safetensors

--- Interactive Generation ---
Enter a prompt to continue (or 'quit' to exit)
Prompt: Sherlock
Response: Sherlock Holmes, who was usually very late in the mornings...
```

## ⚔️ Anexa vs The Giants

Here is how your custom 0.8M parameter model compares to state-of-the-art LLMs. While Anexa is tiny, it beats them in **Privacy** and **Wait Time**.

| Model | Parameters | Training Data | Hardware | Capability | Speed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Anexa** | **0.8 Million** | 1 Book (Sherlock) | 1 Laptop GPU | Basic Completion | **Instant** (Local) |
| **GPT-2 (Small)** | **124 Million** | WebText (8M Docs) | Clusters | Paragraphs | Fast |
| **DeepSeek-V3** | **671 Billion** | Trillions of Tokens | H100 Clusters | Reasoning/Math | Network Lag |
| **Kimi-k2.5** | **hundreds of Bills** | Long-context experts | Massive Clusters | 2M+ Context | Network Lag |
| **Gemini 1.5** | **~Trillions** | Google's Index | TPU Pods | Multimodal | Network Lag |
| **GPT-4** | **~1.8 Trillions** | Entire Internet | Data Centers | Advanced Reasoning | Network Lag |

## 🛠️ Technology Stack

*   **Language**: Python 3.10+
*   **Engine**: PyTorch (CUDA)
*   **Persistence**: SafeTensors (Zero-copy, Safe)
*   **Architecture**: Decoder-only Transformer (GPT-2 style)
    *   Multi-head Self Attention
    *   Feed-Forward Networks
    *   Layer Normalization
    *   Residual Connections

## 🗺️ Roadmap

- [x] Migrate to PyTorch
- [x] Enable GPU Training
- [x] Implement SafeTensors
- [x] Create Documentation
- [ ] Upload to Hugging Face
- [ ] Train on larger dataset (TinyShakespeare)
- [ ] Implement Top-K Sampling

## 📄 License

MIT License. Free to use, modify, and distribute.
