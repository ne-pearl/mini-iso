#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from mini_iso.arrays import Arrays
from mini_iso.clearance import Status, clear_auction
from mini_iso.miscellaneous import DATASETS_ROOT_PATH
from mini_iso.typing import Input, Solution, ZoneId


parser = argparse.ArgumentParser(description="Marginal Units Demo")
parser.add_argument("path", type=Path)
args = parser.parse_args()
inputs: Input = Input.from_json(DATASETS_ROOT_PATH / args.path)

status: Status
solution: Solution | None
status, solution = clear_auction(inputs)
assert status is Status.OPTIMAL
assert solution is not None

# arrays: Arrays = Arrays.init(inputs=inputs, solution=solution)


def check_close(label: str, left: NDArray, right: NDArray, tol: float = 1e-12) -> None:
    scale = np.maximum(np.abs(left), np.abs(right)) + 1.0
    mismatch = np.abs(left - right)
    # print(left)
    # print(right)
    print(label, np.max(mismatch))
    assert np.all(mismatch < tol * scale), mismatch


check_close(
    "objective",
    solution.objective,
    np.dot(solution.offers.offered_price, solution.offers.quantity_dispatched),
)
check_close(
    "balance",
    solution.zones_lines_incidence @ solution.lines.quantity
    + solution.zones_offers_incidence @ solution.offers.quantity_dispatched,
    solution.zones.balance_dual_rhs,
)
check_close(
    "angles",
    solution.base_power * solution.lines.susceptance
    # "-1" for "PL_l = (theta_a = theta_b)/x_ab" where l is (a, b)
    # i.e. "tail - head" rather than "head - tail".
    # See SIII in Fu & Li (2006) "Different Models and Properties on LMP Calculations"
    * (-1 * arrays.zones_lines_incidence.T @ arrays.zones_angle.values),
    arrays.lines_quantity.values,
)
reference_zone: ZoneId = arrays.lines_index.iloc[0][0]
check_close("reference angle", arrays.zones_angle[reference_zone], 0.0)

assert all(arrays.lines_quantity_dual_lb >= 0.0)
assert all(arrays.lines_quantity_dual_ub <= 0.0)
assert all(arrays.offers_dispatched_dual_lb >= 0.0)
assert all(arrays.offers_dispatched_dual_ub <= 0.0)

check_close(
    "strong duality",
    (
        # power_flow_constraints
        np.dot(arrays.zones_price, arrays.zones_load)
        # angle_constraints
        + np.dot(arrays.lines_angles_dual, np.zeros(arrays.lines_index.size))
        # reference_angle_constraint
        + arrays.reference_angle_dual * 0.0
        # power_output_lb
        + np.dot(arrays.offers_dispatched_dual_lb, arrays.offers_quantity)
        # power_output_ub
        - np.dot(arrays.offers_dispatched_dual_ub, np.zeros(arrays.offers_index.size))
        # power_on_line_lb
        + np.dot(arrays.lines_quantity_dual_lb, -1 * arrays.lines_capacity.values)
        # power_on_line_ub
        - np.dot(arrays.lines_quantity_dual_ub, +1 * arrays.lines_capacity.values)
    ),
    arrays.objective,
)
