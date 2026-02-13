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

### ✨ New: Thinking Mode (Experimental)
Anexa now supports **Reasoning Tokens** (`<think>...</think>`), similar to DeepSeek-R1.
*   **How it works:** The model outputs a "Chain of Thought" before the final answer.
*   **Visualization:** In `chat.py`, thoughts are displayed in **Gray** to distinguish them from the answer.
*   **Note:** At 0.8M parameters, the reasoning is currently **structural only** (it mimics the format but makes arithmetic errors).

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

### Detailed Comparison

*   **Anexa (0.8M)**:
    *   **Best For**: Learning how AI works, running locally on old hardware, 100% privacy, zero latency.
    *   **Weakness**: Can only write simple sentences. Cannot code or solve math.
    *   **Analogy**: A bicycle. You built it, you understand it, it's free, and reliable for short trips.

*   **ChatGPT / DeepSeek / Kimi**:
    *   **Best For**: Coding, complex math, summarizing long documents, creative writing.
    *   **Weakness**: Requires internet, privacy concerns (data sent to servers), can be slow (queues).
    *   **Analogy**: A commercial jet. Powerful, fast, gets you across the world, but you don't fly it and you adhere to their schedule.

    *   **Analogy**: A commercial jet. Powerful, fast, gets you across the world, but you don't fly it and you adhere to their schedule.

### 🏆 Benchmark Comparison (Hypothetical)

Here is where the giants stand (and where Anexa is, humbly):

<div align="center">
<table>
<thead>
<tr>
<th align="center">Benchmark</th>
<th align="center"><sup>Qwen3-VL-235B-A22B<br><sup>(Thinking)</sup></sup></th>
<th align="center"><sup>Kimi K2.5<br><sup>(Thinking)</sup></sup></th>
<th align="center"><sup>GPT-5.2 <br><sup>(xhigh)</sup></sup></th>
<th align="center"><sup>Claude 4.5 Opus <br><sup>(Extended Thinking)</sup></sup></th>
<th align="center"><sup>Gemini 3 Pro <br><sup>(High Thinking Level)</sup></sup></th>
<th align="center"><sup>DeepSeek V3.2 <br><sup>(Thinking)</sup></sup></th>
</tr>
</thead>
<tbody>
<tr>
<td align="center" colspan=7><strong>Reasoning &amp; Knowledge</strong></td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">HLE-Full</td>
<td align="center" style="vertical-align: middle">N/A</td>
<td align="center" style="vertical-align: middle">30.1</td>
<td align="center" style="vertical-align: middle">34.5</td>
<td align="center" style="vertical-align: middle">30.8</td>
<td align="center" style="vertical-align: middle">32.2</td>
<td align="center" style="vertical-align: middle">31.5</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">AIME 2025</td>
<td align="center" style="vertical-align: middle">N/A</td>
<td align="center" style="vertical-align: middle">96.1</td>
<td align="center" style="vertical-align: middle">100</td>
<td align="center" style="vertical-align: middle">92.8</td>
<td align="center" style="vertical-align: middle">95.0</td>
<td align="center" style="vertical-align: middle">97.2</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">GPQA-Diamond</td>
<td align="center" style="vertical-align: middle">N/A</td>
<td align="center" style="vertical-align: middle">87.6</td>
<td align="center" style="vertical-align: middle">92.4</td>
<td align="center" style="vertical-align: middle">87.0</td>
<td align="center" style="vertical-align: middle">89.1</td>
<td align="center" style="vertical-align: middle">88.5</td>
</tr>
<tr>
<td align="center" colspan=7><strong>Coding</strong></td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">SWE-Bench Verified</td>
<td align="center" style="vertical-align: middle">N/A</td>
<td align="center" style="vertical-align: middle">76.8</td>
<td align="center" style="vertical-align: middle">80.0</td>
<td align="center" style="vertical-align: middle">80.9</td>
<td align="center" style="vertical-align: middle">78.5</td>
<td align="center" style="vertical-align: middle">79.2</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">LiveCodeBench (v6)</td>
<td align="center" style="vertical-align: middle">N/A</td>
<td align="center" style="vertical-align: middle">85.0</td>
<td align="center" style="vertical-align: middle">-</td>
<td align="center" style="vertical-align: middle">82.2*</td>
<td align="center" style="vertical-align: middle">84.0</td>
<td align="center" style="vertical-align: middle">86.1</td>
</tr>
<tr>
<td align="center" colspan=7><strong>Speed (Tokens / Sec)</strong></td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">Generation Speed</td>
<td align="center" style="vertical-align: middle"><strong>127.9</strong></td>
<td align="center" style="vertical-align: middle">~20</td>
<td align="center" style="vertical-align: middle">~50</td>
<td align="center" style="vertical-align: middle">~30</td>
<td align="center" style="vertical-align: middle">~45</td>
<td align="center" style="vertical-align: middle">~40</td>
</tr>
</tbody>
</table>
</div>

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
