import os
import sys
from itertools import product

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ctc import ctc_loss_and_grad  # noqa: E402


@pytest.fixture
def deterministic_log_probs():
    """Simple, deterministic probability table used across tests."""
    probs = np.array(
        [
            [0.6, 0.25, 0.15],
            [0.2, 0.6, 0.2],
            [0.1, 0.7, 0.2],
            [0.55, 0.3, 0.15],
        ],
        dtype=np.float64,
    )
    return np.log(probs)


def test_ctc_shapes_and_finitedness(deterministic_log_probs):
    labels = [1, 2]
    loss, grad = ctc_loss_and_grad(deterministic_log_probs, labels, blank=0)

    assert np.isfinite(loss)
    assert grad.shape == deterministic_log_probs.shape


def test_ctc_empty_label_sequence_matches_blank_path():
    log_probs = np.log(
        np.array(
            [
                [0.9, 0.1],
                [0.8, 0.2],
                [0.85, 0.15],
            ],
            dtype=np.float64,
        )
    )

    loss, grad = ctc_loss_and_grad(log_probs, labels=[], blank=0)

    expected_loss = -np.sum(log_probs[:, 0])
    assert np.isclose(loss, expected_loss, atol=1e-8)
    assert np.allclose(grad[:, 0], -1.0)
    assert np.allclose(grad[:, 1], 0.0)


def _brute_force_path_prob(log_probs, labels, blank=0):
    """Enumerate all alignments for tiny problems to compute the reference loss."""
    T, C = log_probs.shape
    total_prob = 0.0

    for path in product(range(C), repeat=T):
        # Apply CTC collapse: remove consecutive duplicates, then drop blanks.
        collapsed = []
        prev_symbol = None
        for symbol in path:
            if symbol == blank:
                prev_symbol = None
                continue
            if symbol == prev_symbol:
                continue
            collapsed.append(symbol)
            prev_symbol = symbol

        if collapsed != labels:
            continue

        path_log_prob = sum(log_probs[t, symbol] for t, symbol in enumerate(path))
        total_prob += np.exp(path_log_prob)

    return total_prob


def test_ctc_matches_bruteforce_enumeration(deterministic_log_probs):
    labels = [1]
    log_probs = deterministic_log_probs[:3]

    expected_prob = _brute_force_path_prob(log_probs, labels=labels, blank=0)
    assert expected_prob > 0.0
    loss, _ = ctc_loss_and_grad(log_probs, labels=labels, blank=0)
    assert np.isclose(loss, -np.log(expected_prob), atol=1e-10)


def test_ctc_gradient_matches_finite_difference(deterministic_log_probs):
    rng = np.random.default_rng(seed=42)
    labels = [1, 2]
    log_probs = deterministic_log_probs

    base_loss, grad = ctc_loss_and_grad(log_probs, labels, blank=0)
    direction = rng.normal(size=log_probs.shape)
    epsilon = 1e-5

    loss_plus, _ = ctc_loss_and_grad(log_probs + epsilon * direction, labels, blank=0)
    loss_minus, _ = ctc_loss_and_grad(log_probs - epsilon * direction, labels, blank=0)

    finite_diff_directional = (loss_plus - loss_minus) / (2 * epsilon)
    directional_derivative = np.sum(grad * direction)

    assert np.isclose(directional_derivative, finite_diff_directional, atol=1e-5)
