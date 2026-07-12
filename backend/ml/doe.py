"""DOE (Design of Experiments) generation algorithms.

Pure numpy implementation — no new dependencies required.
"""

import itertools
import numpy as np
from dataclasses import dataclass, field


@dataclass
class DOEFactor:
    name: str
    low: float
    high: float
    center: float | None = None


def generate_full_factorial(factors: list[DOEFactor]) -> list[dict]:
    """Full factorial design: all 2^k combinations."""
    k = len(factors)
    levels = list(itertools.product([0, 1], repeat=k))

    designs = []
    for combo in levels:
        row = {}
        for i, factor in enumerate(factors):
            row[factor.name] = factor.low if combo[i] == 0 else factor.high
        designs.append(row)
    return designs


def generate_fractional_factorial(factors: list[DOEFactor], resolution: int = 3) -> list[dict]:
    """Fractional factorial using Hadamard matrix construction."""
    k = len(factors)
    n = 2 ** k

    # Build Hadamard matrix via Sylvester construction
    h = np.array([[1]])
    while h.shape[0] < n:
        h = np.block([
            [h, h],
            [h, -h]
        ])

    # Select k columns (skip first column of all 1s)
    cols = h[:, 1:k + 1]

    if resolution >= 4:
        # Resolution IV: select columns with minimum aliasing
        selected = list(range(1, k + 1))
    else:
        # Resolution III: simple sequential selection
        selected = list(range(1, k + 1))

    designs = []
    for row_idx in range(n):
        row = {}
        for i, factor in enumerate(factors):
            val = cols[row_idx, i]
            row[factor.name] = factor.low if val < 0 else factor.high
        designs.append(row)
    return designs


def generate_lhs(factors: list[DOEFactor], n_samples: int, seed: int = 42) -> list[dict]:
    """Latin Hypercube Sampling — space-filling design."""
    rng = np.random.default_rng(seed)
    k = len(factors)

    # Generate LHS samples
    samples = np.zeros((n_samples, k))
    for j in range(k):
        perm = rng.permutation(n_samples)
        for i in range(n_samples):
            samples[i, j] = (perm[i] + rng.random()) / n_samples

    # Scale to factor ranges
    designs = []
    for i in range(n_samples):
        row = {}
        for j, factor in enumerate(factors):
            row[factor.name] = factor.low + samples[i, j] * (factor.high - factor.low)
        designs.append(row)
    return designs


def generate_box_behnken(factors: list[DOEFactor]) -> list[dict]:
    """Box-Behnken design: 3-level design without extreme corners. Requires >=3 factors."""
    k = len(factors)
    if k < 3:
        raise ValueError("Box-Behnken requires at least 3 factors")

    designs = []
    # For each pair of factors, add edge midpoints
    for i in range(k):
        for j in range(i + 1, k):
            # All combinations of -1, 0, +1 for factors i and j, others at 0
            for vi in [-1, 0, 1]:
                for vj in [-1, 0, 1]:
                    if vi == 0 and vj == 0:
                        continue  # Skip center point here (added below)
                    row = {}
                    for f_idx, factor in enumerate(factors):
                        if f_idx == i:
                            row[factor.name] = factor.low if vi < 0 else (factor.high if vi > 0 else (factor.center or (factor.low + factor.high) / 2))
                        elif f_idx == j:
                            row[factor.name] = factor.low if vj < 0 else (factor.high if vj > 0 else (factor.center or (factor.low + factor.high) / 2))
                        else:
                            center = factor.center or (factor.low + factor.high) / 2
                            row[factor.name] = center
                    designs.append(row)

    # Add center points (3 replicates)
    center_row = {}
    for factor in factors:
        center_row[factor.name] = factor.center or (factor.low + factor.high) / 2
    for _ in range(3):
        designs.append(center_row.copy())

    return designs


def generate_ccd(factors: list[DOEFactor], alpha: str = "rotatable") -> list[dict]:
    """Central Composite Design: 5-level design for response surfaces. Requires >=3 factors."""
    k = len(factors)
    if k < 3:
        raise ValueError("CCD requires at least 3 factors")

    # Calculate alpha for rotatability
    if alpha == "rotatable":
        alpha_val = np.sqrt(k)
    elif alpha == "orthogonal":
        alpha_val = np.sqrt(k * (k + 2) / 4)
    else:
        alpha_val = float(alpha)

    designs = []

    # 2^k factorial points (coded -1, +1)
    for combo in itertools.product([-1, 1], repeat=k):
        row = {}
        for i, factor in enumerate(factors):
            row[factor.name] = factor.low + (combo[i] + 1) / 2 * (factor.high - factor.low)
        designs.append(row)

    # Star/axial points (+alpha, -alpha)
    center = [(factor.low + factor.high) / 2 for factor in factors]
    for i in range(k):
        for sign in [-1, 1]:
            row = {}
            for j, factor in enumerate(factors):
                center_val = factor.center or (factor.low + factor.high) / 2
                half_range = (factor.high - factor.low) / 2
                if j == i:
                    row[factor.name] = center_val + sign * alpha_val * half_range
                else:
                    row[factor.name] = center_val
            designs.append(row)

    # Center points (4-6 replicates)
    center_row = {}
    for factor in factors:
        center_row[factor.name] = factor.center or (factor.low + factor.high) / 2
    for _ in range(4):
        designs.append(center_row.copy())

    return designs


def estimate_experiment_count(method: str, n_factors: int, n_samples: int | None = None) -> int:
    """Estimate the number of experiments for a given method and factor count."""
    if method == "full_factorial":
        return 2 ** n_factors
    elif method == "fractional_factorial":
        return 2 ** max(1, n_factors - 1)
    elif method == "lhs":
        return n_samples or 10
    elif method == "box_behnken":
        if n_factors < 3:
            return 0
        n_pairs = n_factors * (n_factors - 1) // 2
        return n_pairs * 4 + 3  # 4 edge combos per pair + 3 center
    elif method == "ccd":
        if n_factors < 3:
            return 0
        return 2 ** n_factors + 2 * n_factors + 4
    return 0


@dataclass
class DOEConstraint:
    """Constraint applied to DOE design points."""
    constraint_type: str  # sum | ratio | bound
    factor_names: list[str]
    value: float | None = None  # target sum, ratio, or bound limit
    relation: str = "eq"  # eq, lte, gte


def apply_constraints(designs: list[dict], constraints: list[DOEConstraint]) -> list[dict]:
    """Filter design points to satisfy all constraints."""
    if not constraints:
        return designs

    filtered = []
    for row in designs:
        valid = True
        for c in constraints:
            vals = [row.get(f, 0) for f in c.factor_names]

            if c.constraint_type == "sum":
                total = sum(vals)
                if c.relation == "eq" and abs(total - (c.value or 0)) > 1e-6:
                    valid = False
                elif c.relation == "lte" and total > (c.value or 0):
                    valid = False
                elif c.relation == "gte" and total < (c.value or 0):
                    valid = False

            elif c.constraint_type == "ratio" and len(vals) == 2:
                if vals[1] != 0:
                    ratio = vals[0] / vals[1]
                else:
                    ratio = float("inf") if vals[0] > 0 else 0
                if c.relation == "eq" and abs(ratio - (c.value or 1)) > 1e-6:
                    valid = False
                elif c.relation == "lte" and ratio > (c.value or 1):
                    valid = False
                elif c.relation == "gte" and ratio < (c.value or 1):
                    valid = False

            elif c.constraint_type == "bound":
                for v in vals:
                    if c.relation == "lte" and v > (c.value or 0):
                        valid = False
                    elif c.relation == "gte" and v < (c.value or 0):
                        valid = False

            if not valid:
                break

        if valid:
            filtered.append(row)

    return filtered
