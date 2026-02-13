# Anexa: Old Architecture (Pure Python)

This document describes the **original, educational version** of Anexa (formerly MicroGPT). This version was designed to be a "from scratch" implementation of a Transformer, with zero dependencies other than the standard Python library.

## Core Components

### 1. The `Value` Class
The heart of this version was the `Value` class, a custom Autograd (Automatic Differentiation) engine inspired by Andrej Karpathy's typical `micrograd`.

-   **Purpose**: To track every mathematical operation (+, -, *, /) and build a dynamic computation graph.
-   **Mechanism**:
    -   Each `Value` object stored a scalar `data` and a gradient `grad`.
    -   It maintained pointers to its `_children` (parameters it was calculated from).
    -   It implemented `backward()` to perform backpropagation using the chain rule.

```python
class Value:
    def __init__(self, data, _children=()):
        self.data = data
        self.grad = 0
        self._prev = set(_children)
        self._backward = lambda: None
```

### 2. Manual Matrix Multiplication
Since we didn't use NumPy or PyTorch, all matrix operations were written as nested loops.

```python
def matmul(a, b):
    # C[i][j] = sum(A[i][k] * B[k][j])
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) 
             for j in range(len(b[0]))] 
            for i in range(len(a))]
```

### 3. Sequence Parallelism
We implemented "Sequence Parallelism" manually by refactoring the `gpt` function.
-   **Original**: Processed 1 token at a time.
-   **Improved**: Processed a list of `N` tokens in parallel using list comprehensions and helper functions (`transpose`, `causal_mask`).

### Performance
-   **Training Speed**: Extremely slow (~1 iteration per second on CPU).
-   **Optimization**: Limited by Python's interpreter overhead for millions of small objects.

## Why We Moved
While excellent for understanding *how* neural networks work, this architecture was too slow for training on anything larger than a few sentences.
