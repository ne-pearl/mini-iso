import numpy as np
import pandas as pd
from demo_marginal import (
    TOL,
    inputs,
    is_basic_offers,
    is_marginal,
    lambda_lines,
    lambda_ref,
    lambda_zones,
    num_zones,
    offer_zone_price,
    solution,
)


def check_zero(message: str, x: np.ndarray) -> None:
    assert np.linalg.norm(x, ord=np.inf) <= TOL, message


check_zero(
    "KKT stationarity for lines quantity",
    solution.zones_lines_incidence.T @ lambda_zones
    + lambda_lines * (-1)
    + solution.lines.quantity_lb_coef
    + solution.lines.quantity_ub_coef,
)

check_zero(
    "KKT stationarity for zones angle",
    (solution.base_power * -1 * solution.zones_lines_incidence)
    @ (inputs.lines.susceptance * lambda_lines)
    + solution.zones.angle_lb_coef
    + solution.zones.angle_ub_coef
    + lambda_ref * np.eye(num_zones, 1).flatten(),
)

check_zero(
    "KKT stationarity for offers quantity",
    solution.offers.offered_price
    - (
        solution.zones_offers_incidence.T @ lambda_zones
        + solution.offers.quantity_lb_coef
        + solution.offers.quantity_ub_coef
    ),
)

assert all(solution.offers.quantity_lb_coef[is_basic_offers] == 0)
assert all(solution.offers.quantity_ub_coef[is_basic_offers] == 0)

x_lst, *_ = np.linalg.lstsq(
    solution.zones_offers_incidence[:, is_basic_offers].T.todense(),
    solution.offers.offered_price[is_basic_offers],
    rcond=None,
)

offers_zone = pd.Series(
    inputs.generators.zone.loc[inputs.offers.index.get_level_values("generator")].values,
    index=inputs.offers.index,
    name=inputs.generators.zone.name,
)
temporary = pd.concat((is_basic_offers, offers_zone), axis=1)
zones_marginal = temporary.groupby(offers_zone.name)[is_basic_offers.name].aggregate(any)
