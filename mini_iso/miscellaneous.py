from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Final, Literal
from bokeh.models.widgets.tables import (
    CellFormatter,
    NumberFormatter,
)
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import panel as pn
import param as pm
from panel.widgets import Tabulator

ADDRESS: Final[str] = "*"
PORT: Final[int] = 5008
DATASETS_ROOT_PATH: Final[Path] = Path(__file__).parent.parent
assert DATASETS_ROOT_PATH.exists()
assert DATASETS_ROOT_PATH.is_dir()

# Tolerance for detection of binding constraints
BIND_TOL: Final[float] = 1.0 / 100.0

# For panel.widgets.indicators; default font is too large
FONT_SIZE: Final = 15
INDICATOR_FONT_SIZES: Final[dict[str, str]] = dict(
    font_size=str(FONT_SIZE),
    title_size=str(FONT_SIZE),
)

# Markdown header levels
MARKDOWN_LEVEL_UPPER: Final = 2
MARKDOWN_LEVEL_LOWER: Final = 3


def digits_key(text: str) -> float:
    """A sort key for strings with embedded numbers."""
    matched: re.Match | None = re.search(r"(\d+)", text)
    if matched is None:
        return np.inf  # lowest-priority when sorted
    return float(matched.group(0))


def index_digits_key(index: pd.Index) -> pd.Index:
    """For sorting dataframe index labels with embedded numbers."""
    return index.map(digits_key)


@dataclass(frozen=True, slots=True)
class Format:
    align: Literal["center", "left", "right"]
    formatter: dict[str, bool | int | str] | CellFormatter

    @classmethod
    def from_unit(cls, unit: str, precision: int = 2) -> Format:
        return cls(
            align="right",
            formatter={
                "precision": precision,
                "symbol": " " + unit.strip(),
                "symbolAfter": True,
                "type": "money",  # hijack money for physical units
            },
        )

    @classmethod
    def boolean(cls, allow_empty: bool = False, allow_truthy: bool = False) -> Format:
        return cls(
            align="center",
            formatter={
                "allowEmpty": allow_empty,
                "allowTruthy": allow_truthy,
                "type": "tickCross",
            },
        )


admittance_siemens = Format.from_unit("S", precision=1)
boolean_check = Format.boolean(allow_empty=False)
tristate_check = Format.boolean(allow_empty=True)
fraction_percentage = Format(align="right", formatter=NumberFormatter(format="0%"))
power_megawatts = Format.from_unit("MW")
price_usd_per_mwh = Format.from_unit("$/MHh")
payment_usd_per_h = Format.from_unit("$/h")
real_unspecified = Format(align="right", formatter=NumberFormatter(format="0.0"))


def filter_columns(tabulator: Tabulator, columns: list[str]) -> Tabulator:
    hidden_columns: set[str] = set(tabulator.value.columns).difference(columns)
    tabulator.hidden_columns = list(hidden_columns)
    return tabulator


def labeled(
    viewable: pn.viewable.Viewable,
    label: str | None = None,
    level: int = 2,
) -> pn.Column:
    assert level - 1 in range(6)
    name_: str | None = label or getattr(viewable, "name", getattr(viewable, "label"))
    assert name_ is not None
    return pn.Column(
        pn.pane.Markdown(f"{'#' * level} {label}"),
        viewable,
    )


def tabulator_item(
    param: pm.Parameter,
    name: str | None = None,
    show_columns: list[str] | None = None,
    disabled: bool = True,
    **kwargs,
) -> tuple[str, Tabulator]:
    name_: str = name or param.label
    assert name_ is not None

    tabulator: Tabulator = Tabulator.from_param(
        param,
        name=name_,
        disabled=disabled,
        **kwargs,
    )

    # Compute hidden columns from list of columns to show
    all_columns: list[str] = tabulator.value.columns.values.tolist()
    if show_columns is None:
        show_columns = all_columns
    tabulator.hidden_columns = list(set(all_columns).difference(show_columns))

    return name_, tabulator


def get_series(dataframe: pd.DataFrame, name: str) -> pd.Series:
    if name in dataframe.index.names:
        data: NDArray = dataframe.index.get_level_values(level=name)
        return pd.Series(data=data, index=dataframe.index)
    return dataframe[name]
