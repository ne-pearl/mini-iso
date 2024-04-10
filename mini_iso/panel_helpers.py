from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, TypeAlias
from bokeh.models.widgets.tables import (
    CellFormatter,
    NumberFormatter,
    BooleanFormatter,
)
import panel as pn
import param as pm
from panel.widgets import Tabulator


@dataclass(frozen=True, slots=True)
class Format:
    align: Literal["center", "left", "right"]
    formatter: dict[str, bool | int | str] | CellFormatter

    @classmethod
    def from_unit(cls, unit: str, precision: int = 0) -> Format:
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


admittance_siemens = Format.from_unit("S")
boolean_check = Format.boolean(allow_empty=False)
tristate_check = Format.boolean(allow_empty=True)
fraction_percentage = Format(align="right", formatter=NumberFormatter(format="0%"))
money_dollars = Format(align="right", formatter=NumberFormatter(format="$0"))
power_megawatts = Format.from_unit("MW")
real_unspecified = Format(align="right", formatter=NumberFormatter(format="0.0"))


def filter_columns(tabulator: Tabulator, columns: list[str]) -> Tabulator:
    hidden_columns: set[str] = set(tabulator.value.columns).difference(columns)
    tabulator.hidden_columns = list(hidden_columns)
    return tabulator


def labeled(tabulator: Tabulator, name: str | None = None, level: int = 3) -> pn.Column:
    assert level - 1 in range(6)
    name_: str | None = name or tabulator.name
    assert name_ is not None
    return pn.Column(
        pn.pane.Markdown(f"{'#' * level} {name}"),
        tabulator,
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
