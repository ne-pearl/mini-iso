#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Final
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pandera.typing import Series
from mini_iso.clearance import BasisStatus, Solution, Status, clear_auction
from mini_iso.miscellaneous import DATASETS_ROOT_PATH
from mini_iso.typing_ import (
    Input,
    Lines,
    LinesSolution,
    OffersSolution,
    PriceUSDPerMWh,
    ZoneId,
    ZonesSolution,
)

TOL: Final = 1e-6

pd.options.display.max_columns = None

parser = argparse.ArgumentParser(description="Marginal Units Demo")
parser.add_argument(
    "--path",
    type=Path,
    default="mini_iso/datasets/mini_new_england/mini_new_england.json",
)
args = parser.parse_args()
inputs: Input = Input.from_json(DATASETS_ROOT_PATH / args.path)

status: Status
solution: Solution | None
status, solution = clear_auction(inputs)
assert status is Status.OPTIMAL
assert solution is not None

# arrays: Arrays = Arrays.init(inputs=inputs, solution=solution)
num_lines: Final[int] = solution.lines.shape[0]
num_offers: Final[int] = solution.offers.shape[0]
num_zones: Final[int] = solution.zones.shape[0]


def check_close(label: str, left: NDArray, right: NDArray, tol: float = TOL) -> None:
    scale = np.maximum(np.abs(left), np.abs(right)) + 1.0
    mismatch = np.abs(left - right)
    print(label, np.max(mismatch))
    assert np.all(mismatch < tol * scale), mismatch


reference_zone: ZoneId = inputs.lines[Lines.zone_from].iat[0]

balance_residual = (
    solution.zones_lines_incidence @ solution.lines.quantity
    + solution.zones_offers_incidence @ solution.offers.quantity_dispatched
    - solution.zones.balance_rhs
)

angles_residual = (
    solution.base_power * inputs.lines.susceptance
    # "-1" for "PL_l = (theta_a - theta_b)/x_ab" where l is (a, b)
    # i.e. "tail - head" rather than "head - tail".
    # See SIII in Fu & Li (2006) "Different Models and Properties on LMP Calculations"
    * (-1 * solution.zones_lines_incidence.T @ solution.zones.angle)
    - solution.lines.quantity
)

check_close(
    "objective",
    np.dot(solution.offers.offered_price, solution.offers.quantity_dispatched),
    solution.objective,
)
check_close(
    "balance",
    balance_residual,
    np.zeros(num_zones),
)
check_close(
    "angles",
    angles_residual,
    np.zeros(num_lines),
)
check_close(
    "reference angle",
    solution.zones.angle.at[reference_zone],
    0.0,
)

assert all(solution.lines.quantity_lb_coef >= 0.0)
assert all(solution.lines.quantity_ub_coef <= 0.0)
assert all(solution.offers.quantity_lb_coef >= 0.0)
assert all(solution.offers.quantity_ub_coef <= 0.0)

lambda_lines = solution.lines.angle_coef
lambda_zones = solution.zones.balance_coef
lambda_ref = solution.reference_angle_coef

check_close(
    "strong duality",
    (
        # balance constraint
        sum(solution.zones.price * inputs.zones.load)
        # reference angle
        + lambda_ref * solution.reference_angle_rhs
        # angle constraints
        + sum(lambda_lines * solution.lines.angle_rhs)
        # generator capacity bounds
        + sum(solution.offers.quantity_lb_coef * solution.offers.quantity_lb_rhs)
        + sum(solution.offers.quantity_ub_coef * solution.offers.quantity_ub_rhs)
        # line capacity bounds
        + sum(solution.lines.quantity_lb_coef * solution.lines.quantity_lb_rhs)
        + sum(solution.lines.quantity_ub_coef * solution.lines.quantity_ub_rhs)
    ),
    solution.objective,
)

check_close(
    "KKT stationarity for lines quantity",
    (
        solution.zones_lines_incidence.T @ lambda_zones
        + lambda_lines * (-1)
        + solution.lines.quantity_lb_coef
        + solution.lines.quantity_ub_coef
    ),
    np.zeros(num_lines),
)

check_close(
    "KKT stationarity for offers quantity",
    (
        solution.offers.offered_price
        - (
            solution.zones_offers_incidence.T @ lambda_zones
            + solution.offers.quantity_lb_coef
            + solution.offers.quantity_ub_coef
        )
    ),
    np.zeros(num_offers),
)

check_close(
    "KKT stationarity for zones angle",
    (
        (solution.base_power * -1 * solution.zones_lines_incidence)
        @ (inputs.lines.susceptance * lambda_lines)
        + solution.zones.angle_lb_coef
        + solution.zones.angle_ub_coef
        + lambda_ref * np.eye(num_zones, 1).flatten()
    ),
    np.zeros(num_zones),
)

generators_zone_price = Series[PriceUSDPerMWh](
    data=solution.zones.price.loc[inputs.generators.zone].values,
    index=inputs.generators.index,
    name="lmp",
)

offer_zone_price = Series[PriceUSDPerMWh](
    data=generators_zone_price.loc[
        solution.offers.index.get_level_values("generator")
    ].values,
    index=solution.offers.index,
    name="offer_zone_price",
)

is_marginal = (offer_zone_price - inputs.offers.price).abs() < TOL

print()
print("offers.quantity:")
print("================")
print(
    pd.concat(
        (
            is_marginal,
            solution.offers[
                [
                    OffersSolution.quantity_basis,
                    OffersSolution.quantity_lb_coef,
                    OffersSolution.quantity_ub_coef,
                    OffersSolution.quantity_lb_rhs,
                    OffersSolution.quantity_ub_rhs,
                ]
            ],
        ),
        axis=1,
    )
)

print()
print("lines.quantity:")
print("===============")
print(
    solution.lines[
        [
            LinesSolution.quantity_basis,
            LinesSolution.quantity_lb_coef,
            LinesSolution.quantity_ub_coef,
            LinesSolution.quantity_lb_rhs,
            LinesSolution.quantity_ub_rhs,
        ]
    ]
)

print()
print("zones.angle:")
print("============")
print(
    solution.zones[
        [
            ZonesSolution.angle,
            ZonesSolution.angle_basis,
            ZonesSolution.angle_lb_coef,
            ZonesSolution.angle_lb_rhs,
            ZonesSolution.angle_ub_coef,
            ZonesSolution.angle_ub_rhs,
        ]
    ]
)

a_balance = np.hstack(
    (
        solution.zones_lines_incidence.todense(),
        solution.zones_offers_incidence.todense(),
        np.zeros((num_zones, num_zones)),
    )
)
b_balance = solution.zones.balance_rhs.values

e = np.zeros((1, num_zones))
(column_indices,) = (solution.zones.index == reference_zone).nonzero()
assert column_indices.size == 1
e[0, column_indices] = 1.0
a_angles = np.vstack(
    (
        np.hstack(
            (
                -np.eye(num_lines),
                np.zeros((num_lines, num_offers)),
                np.diag(-1 * solution.base_power * inputs.lines.susceptance)
                @ solution.zones_lines_incidence.T,
            )
        ),
        np.hstack(
            (
                np.zeros((1, num_lines + num_offers)),
                e,
            )
        ),
    )
)
b_angles = np.zeros(num_lines + 1)

c_offers = solution.offers.offered_price
c_all = np.hstack((np.zeros(num_lines), c_offers, np.zeros(num_zones)))

x_lines = solution.lines.quantity
x_offers = solution.offers.quantity_dispatched
x_zones = solution.zones.angle
x_all = np.hstack((x_lines, x_offers, x_zones))

is_basic_lines = solution.lines.quantity_basis == BasisStatus.BASIC
is_basic_offers = solution.offers.quantity_basis == BasisStatus.BASIC
is_basic_zones = solution.zones.angle_basis == BasisStatus.BASIC
is_basic_all = np.hstack((is_basic_lines, is_basic_offers, is_basic_zones))

assert np.linalg.norm(a_balance @ x_all - b_balance, ord=np.inf) < TOL
assert np.linalg.norm(a_angles @ x_all - b_angles, ord=np.inf) < TOL

num_equalities: Final[int] = a_balance.shape[0] + a_angles.shape[0]
num_basic: Final[int] = is_basic_all.sum()
print(f"     equalities: {num_equalities}")
print(f"basic variables: {num_basic}")

a_all = np.vstack((a_balance, a_angles))
b_all = np.hstack((b_balance, b_angles))


a_basic = a_all[:, is_basic_all]
a_nonbasic = a_all[:, ~is_basic_all]
x_nonbasic = x_all[~is_basic_all]
x_basic = np.linalg.solve(a_basic, b_all - a_nonbasic @ x_nonbasic)

assert np.linalg.norm(x_all[is_basic_all] - x_basic, ord=np.inf) < TOL
