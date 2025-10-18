import numpy as np
from math import log, exp

def log_sum_exp(a, b):
    """
    Numerically stable log-sum-exp for two values, handling -∞ inputs properly.
    """
    # If both are -∞, return -∞ instead of producing NaN
    if a == -np.inf and b == -np.inf:
        return -np.inf
    # Otherwise do the usual stable log-sum-exp
    if a > b:
        return a + log(1 + exp(b - a))
    else:
        return b + log(1 + exp(a - b))

def ctc_loss_and_grad(log_probs, labels, blank=0):
    """
    Computes the CTC loss and gradient for a single (log_probs, labels) pair.
    
    Parameters
    ----------
    log_probs : np.ndarray of shape (T, C)
        log_probs[t, c] = log probability of class c at time t.
        - T = number of time steps
        - C = number of classes (including blank)
    labels : list or np.ndarray of shape (M,)
        The ground-truth label sequence (no blanks). Example: [3, 5, 5].
    blank : int
        Index of the blank symbol.

    Returns
    -------
    loss : float
        The negative log-likelihood (CTC loss).
    grad : np.ndarray of shape (T, C)
        The gradient of the loss w.r.t. log_probs[t, c].
    """
    T, C = log_probs.shape
    M = len(labels)

    # Optional :If T < 2*M + 1, no valid alignment is possible => loss = ∞
    # if T < 2 * M + 1:
    #     return float('inf'), np.zeros_like(log_probs)

    # Build the extended label sequence: [blank, l1, blank, l2, ..., blank, lM, blank]
    ext_labels = []
    for lab in labels:
        ext_labels.append(blank)
        ext_labels.append(lab)
    ext_labels.append(blank)
    L = len(ext_labels)  # should be 2*M + 1

    # Initialize alpha (forward) and beta (backward) in log-space
    alpha = -np.inf * np.ones((T, L), dtype=np.float64)
    beta  = -np.inf * np.ones((T, L), dtype=np.float64)

    # alpha(0, 0) = log_probs(0, blank)
    alpha[0, 0] = log_probs[0, blank]
    # alpha(0, 1) = log_probs(0, ext_labels[1]) if it exists
    if L > 1:
        alpha[0, 1] = log_probs[0, ext_labels[1]]

    # Forward Pass (alpha)
    for t in range(1, T):
        for s in range(L):
            c = ext_labels[s]

            # Possible previous alpha values
            log_sum = alpha[t-1, s]  # same symbol

            if s - 1 >= 0:
                log_sum = log_sum_exp(log_sum, alpha[t-1, s-1])  # neighbor

            if s - 2 >= 0 and ext_labels[s] != ext_labels[s-2]:
                log_sum = log_sum_exp(log_sum, alpha[t-1, s-2])  # skip-1 if not repeating symbol

            alpha[t, s] = log_probs[t, c] + log_sum

    # Backward Pass (beta)
    # Initialize last row
    beta[T-1, L-1] = log_probs[T-1, ext_labels[L-1]]
    if L > 1:
        beta[T-1, L-2] = log_probs[T-1, ext_labels[L-2]]

    # Fill the last row going left
    for s in range(L-3, -1, -1):
        beta[T-1, s] = log_probs[T-1, ext_labels[s]] + \
                       log_sum_exp(beta[T-1, s+1], beta[T-1, s])

    # Now move backward in time
    for t in range(T-2, -1, -1):
        for s in range(L-1, -1, -1):
            c = ext_labels[s]
            log_sum = beta[t+1, s]

            if s + 1 < L:
                log_sum = log_sum_exp(log_sum, beta[t+1, s+1])

            if s + 2 < L and ext_labels[s] != ext_labels[s+2]:
                log_sum = log_sum_exp(log_sum, beta[t+1, s+2])

            beta[t, s] = log_probs[t, c] + log_sum

    # Total Log-Likelihood
    if L == 1:
        # Edge case: label sequence is empty => only [blank]
        log_prob = alpha[T-1, 0]
    else:
        log_prob = log_sum_exp(alpha[T-1, L-1], alpha[T-1, L-2])

    loss = -log_prob  # negative log-likelihood

    # Gradient Computation
    grad = np.zeros_like(log_probs)

    for t in range(T):
        for s in range(L):
            c = ext_labels[s]
            gamma_ts = alpha[t, s] + beta[t, s] - log_prob
            p = np.exp(gamma_ts)  # posterior
            grad[t, c] -= p

    return loss, grad