import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from ctc import ctc_loss_and_grad

def test_ctc_basic():
    T, C = 5, 4
    labels = [1, 2]
    raw_logits = np.random.randn(T, C)
    log_probs = raw_logits - np.logaddexp.reduce(raw_logits, axis=1, keepdims=True)

    loss, grad = ctc_loss_and_grad(log_probs, labels, blank=0)
    assert np.isfinite(loss), "Loss should be finite"
    assert grad.shape == (T, C), "Gradient shape mismatch"

def test_ctc():
    np.random.seed(0)
    
    T = 5   # time steps
    C = 4   # classes (0=blank, plus 3 real labels)
    labels = [1, 2]  # ground truth (no blanks)
    
    # random log-probabilities
    raw_logits = np.random.randn(T, C)
    # Convert to log_probs via log_softmax
    # quick approximate log_softmax
    log_probs = raw_logits - np.logaddexp.reduce(raw_logits, axis=1, keepdims=True)
    
    loss, grad = ctc_loss_and_grad(log_probs, labels, blank=0)
    
    print("CTC Loss:", loss)
    print("Gradient shape:", grad.shape)
    print("Gradient sample:\n", grad)