from typing import Sequence
import math

def prob_rain_more_than_n(p: Sequence[float], n: int) -> float:
    """
    Returns the probability that the number of rainy days is at least n.
    p: Sequence of daily rain probabilities (length = 365).
    n: Threshold number of rainy days.
    """
    num_days = len(p)
    
    # dp[i][k] = probability of exactly k rainy days among the first i days
    dp = [[0.0] * (num_days + 1) for _ in range(num_days + 1)]
    
    # Base case: probability of 0 rainy days out of 0 is 1
    dp[0][0] = 1.0
    
    # Fill the DP table
    for i in range(1, num_days + 1):
        prob_rain = p[i - 1]
        for k in range(i + 1):
            # Case 1: day i is not rainy
            dp[i][k] = dp[i - 1][k] * (1 - prob_rain)
            # Case 2: day i is rainy
            if k > 0:
                dp[i][k] += dp[i - 1][k - 1] * prob_rain
    
    # Sum the probabilities of having at least n rainy days
    return sum(dp[num_days][k] for k in range(n, num_days + 1))

# Tests

def test_prob_rain_more_than_n(func):
    test_cases = [
        # All zeros
        ([0.0, 0.0, 0.0], 0, 1.0),
        ([0.0, 0.0, 0.0], 1, 0.0),

        # All ones
        ([1.0, 1.0, 1.0], 1, 1.0),
        ([1.0, 1.0, 1.0], 3, 1.0),

        # All 0.5
        ([0.5, 0.5, 0.5], 2, 0.5),

        # Mixed (3 days)
        ([0.2, 0.3, 0.4], 1, 0.664),
        ([0.2, 0.3, 0.4], 2, 0.212),

        # Mixed (2 days)
        ([0.3, 0.7], 2, 0.21),
    ]

    for p, n, expected in test_cases:
        result = func(p, n)
        # Compare within a small tolerance
        if not math.isclose(result, expected, abs_tol=1e-6):
            print(f"Test failed for p={p}, n={n}. "
                  f"Expected {expected}, got {result}")
        else:
            print(f"Test passed for p={p}, n={n} => {result}")

test_prob_rain_more_than_n(prob_rain_more_than_n)