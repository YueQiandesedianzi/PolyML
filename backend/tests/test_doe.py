import numpy as np

from ml.doe import (
    DOEConstraint,
    DOEFactor,
    apply_constraints,
    generate_box_behnken,
    generate_ccd,
    generate_fractional_factorial,
    generate_full_factorial,
)


def _factors(n):
    return [DOEFactor(f"x{i}", -1.0, 1.0, 0.0) for i in range(n)]


def test_standard_design_counts_and_centers():
    assert len(generate_full_factorial(_factors(3))) == 8
    assert len(generate_fractional_factorial(_factors(6), resolution=3)) == 8
    bbd = generate_box_behnken(_factors(3))
    assert len(bbd) == 15
    assert sum(all(value == 0 for value in row.values()) for row in bbd) == 3
    ccd = generate_ccd(_factors(3))
    assert len(ccd) == 18


def test_constraints_filter_expected_rows():
    rows = generate_full_factorial([DOEFactor("a", 0, 1), DOEFactor("b", 0, 1)])
    filtered = apply_constraints(rows, [DOEConstraint("sum", ["a", "b"], 1.0, "eq")])
    assert {(row["a"], row["b"]) for row in filtered} == {(1.0, 0.0), (0.0, 1.0)}
