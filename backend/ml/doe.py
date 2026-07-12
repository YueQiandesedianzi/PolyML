"""DOE (Design of Experiments) generation algorithms.

Pure numpy implementation — no new dependencies required.
"""

import itertools
import numpy as np
from dataclasses import dataclass, field
from pyDOE3 import bbdesign, ccdesign, ff2n, fracfact_by_res, lhs


@dataclass
class DOEFactor:
    name: str
    low: float
    high: float
    center: float | None = None


def generate_full_factorial(factors: list[DOEFactor]) -> list[dict]:
    """Full factorial design: all 2^k combinations."""
    return _decode_coded(ff2n(len(factors)), factors)


def generate_fractional_factorial(factors: list[DOEFactor], resolution: int = 3) -> list[dict]:
    """Minimum-run two-level fractional factorial at the requested resolution."""
    if resolution not in {3, 4, 5}:
        raise ValueError("Fractional factorial resolution must be 3, 4, or 5")
    return _decode_coded(fracfact_by_res(len(factors), resolution), factors)


def generate_lhs(factors: list[DOEFactor], n_samples: int, seed: int = 42) -> list[dict]:
    """Latin Hypercube Sampling — space-filling design."""
    if n_samples < 1:
        raise ValueError("LHS requires at least one sample")
    samples = lhs(len(factors), samples=n_samples, criterion="maximin", random_state=seed)

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

    return _decode_coded(bbdesign(k, center=3), factors)


def generate_ccd(factors: list[DOEFactor], alpha: str = "rotatable") -> list[dict]:
    """Central Composite Design: 5-level design for response surfaces. Requires >=3 factors."""
    k = len(factors)
    if k < 3:
        raise ValueError("CCD requires at least 3 factors")

    alpha_mode = "r" if alpha == "rotatable" else ("o" if alpha == "orthogonal" else alpha)
    return _decode_coded(ccdesign(k, center=(2, 2), alpha=alpha_mode, face="ccc"), factors)


def _decode_coded(matrix: np.ndarray, factors: list[DOEFactor]) -> list[dict]:
    """Map coded DOE levels to physical factor values, preserving a zero center."""
    designs = []
    for coded_row in np.asarray(matrix, dtype=float):
        row = {}
        for value, factor in zip(coded_row, factors):
            center = factor.center if factor.center is not None else (factor.low + factor.high) / 2
            half_range = (factor.high - factor.low) / 2
            row[factor.name] = float(center + value * half_range)
        designs.append(row)
    return designs


def estimate_experiment_count(method: str, n_factors: int, n_samples: int | None = None) -> int:
    """Estimate the number of experiments for a given method and factor count."""
    if method == "full_factorial":
        return 2 ** n_factors
    elif method == "fractional_factorial":
        return len(fracfact_by_res(n_factors, 3))
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
