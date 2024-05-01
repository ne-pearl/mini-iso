#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pandera.typing import Series
import scipy
from mini_iso.arrays import Arrays
from mini_iso.clearance import BasisStatus, Status, clear_auction
from mini_iso.miscellaneous import DATASETS_ROOT_PATH
from mini_iso.typing_ import (
    Input,
    Lines,
    LinesSolution,
    OffersSolution,
    PriceUSDPerMWh,
    Zones,
    ZonesSolution,
)

pd.options.display.max_columns = None

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


def check_close(label: str, left: NDArray, right: NDArray, tol: float = 1e-10) -> None:
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
    solution.objective,
    np.dot(solution.offers.offered_price, solution.offers.quantity_dispatched),
)
check_close(
    "balance",
    balance_residual,
    np.zeros(solution.zones.balance_rhs.shape),
)
check_close("angles", angles_residual, np.zeros(solution.lines.quantity.shape))
check_close(
    "reference angle",
    solution.zones.at[reference_zone, ZonesSolution.angle],
    0.0,
)

assert all(solution.lines.quantity_lb_coef >= 0.0)
assert all(solution.lines.quantity_ub_coef <= 0.0)
assert all(solution.offers.quantity_lb_coef >= 0.0)
assert all(solution.offers.quantity_ub_coef <= 0.0)

check_close(
    "strong duality",
    (
        # balance constraint
        sum(solution.zones.price * inputs.zones.load)
        # reference angle
        + solution.reference_angle_coef * solution.reference_angle_rhs
        # angle constraints
        + sum(solution.lines.angle_coef * solution.lines.angle_rhs)
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
        solution.zones_lines_incidence.T @ solution.zones.balance_coef
        + solution.lines.angle_coef * (-1)
        + solution.lines.quantity_lb_coef
        + solution.lines.quantity_ub_coef
    ),
    np.zeros(solution.lines.index.size),
)

check_close(
    "KKT stationarity for offers quantity",
    (
        solution.offers.offered_price
        - (
            solution.zones_offers_incidence.T @ solution.zones.balance_coef
            + solution.offers.quantity_lb_coef
            + solution.offers.quantity_ub_coef
        )
    ),
    np.zeros(solution.offers.index.size),
)

check_close(
    "KKT stationarity for zones angle",
    (
        (solution.base_power * -1 * solution.zones_lines_incidence)
        @ (inputs.lines.susceptance * solution.lines.angle_coef)
        + solution.zones.angle_lb_coef
        + solution.zones.angle_ub_coef
        + solution.reference_angle_coef * np.eye(solution.zones.index.size, 1).flatten()
    ),
    np.zeros(solution.zones.index.size),
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
    name="is_marginal",
)

is_marginal = (offer_zone_price - inputs.offers.price).abs() <= 1e-6
temporary = pd.concat((solution.offers.quantity_basis.map(lambda e: e.name), is_marginal), axis=1)
print(temporary)

basic, = (solution.offers.quantity_basis == BasisStatus.BASIC).values.nonzero()

print("basic indices:", basic)