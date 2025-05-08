# ctc-algorithms

A lightweight Python package for computing Connectionist Temporal Classification (CTC) loss and gradients—commonly used in speech‐recognition and sequence‐to‐sequence models.

## 🚀 Features

- **Numerically stable** log‐sum‐exp implementation  
- Forward–backward (α/β) dynamic‐programming to compute CTC loss  
- Gradient computation for backpropagation  
- Simple API: one function, zero external dependencies beyond NumPy  
- Built‐in test harness with `pytest`

## 📦 Installation

```bash
# Clone (or download) then from the project root:
pip install -e .
```

## Usage

```bash
import numpy as np
from ctc import ctc_loss_and_grad

# Example: T=5 time‐steps, C=4 classes (class 0 = blank)
T, C = 5, 4
labels = [1, 2]  # ground‐truth (no blanks)

# Create random log‐probabilities via log‐softmax
raw = np.random.randn(T, C)
log_probs = raw - np.logaddexp.reduce(raw, axis=1, keepdims=True)

loss, grad = ctc_loss_and_grad(log_probs, labels, blank=0)
print("CTC loss:", loss)
print("Gradient shape:", grad.shape)
```