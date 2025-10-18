"""Minimal example demonstrating the CTC loss/gradient API."""
import numpy as np

from ctc import ctc_loss_and_grad


def make_log_probs(time_steps: int, num_classes: int, seed: int = 0) -> np.ndarray:
    """Produce log-probabilities via log-softmax with a fixed seed for reproducibility."""
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(time_steps, num_classes))
    return raw - np.logaddexp.reduce(raw, axis=1, keepdims=True)


def main() -> None:
    # Example with T time-steps and C classes (class 0 = blank).
    time_steps, num_classes = 6, 4
    labels = [1, 2, 1]  # ground-truth sequence without blanks

    log_probs = make_log_probs(time_steps, num_classes)
    loss, grad = ctc_loss_and_grad(log_probs, labels, blank=0)

    print(f"CTC loss: {loss:.6f}")
    print("Gradient summary:")
    print(f"  shape={grad.shape}, mean={grad.mean():.6f}, std={grad.std():.6f}")


if __name__ == "__main__":
    main()
