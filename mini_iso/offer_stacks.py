from __future__ import annotations
import dataclasses
import enum
import pathlib
import typing
import altair as alt
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pandera import DataFrameModel, Field
from pandera.typing import DataFrame, Series
from mini_iso.miscellaneous import BIND_TOL
from mini_iso.typing import (
    OFFERS_INDEX_LABELS,
    Fraction,
    Generators,
    GeneratorId,
    Input,
    Lines,
    LinesFlow,
    PriceUSDPerMWh,
    Offers,
    OffersDispatched,
    PowerMW,
    ZoneId,
    Zones,
    ZonesPrice,
)
from mini_iso.datasets.mini_new_england import load_system

INFLOW_NAME: typing.Final[str] = "INFLOW"
OUTFLOW_NAME: typing.Final[str] = "OUTFLOW"
FLOW_PRICE: typing.Final[str] = "flow_price"

LOAD_KEY: typing.Final[str] = "Load"
PRICE_LOCAL_KEY: typing.Final[str] = "Local Price"


class Utilization(enum.Enum):
    COMPLETE = "complete"
    IMPORTED = "imported"
    MARGINAL = "marginal"
    UNUSED = "unused"


class OfferStack(DataFrameModel):
    """Schema for offer data."""

    generator: Series[str]
    tranche: Series[str]
    capacity_left: Series[PowerMW] = Field(coerce=True)
    capacity_right: Series[PowerMW] = Field(coerce=True)
    dispatched_left: Series[PowerMW] = Field(coerce=True)
    dispatched_right: Series[PowerMW] = Field(coerce=True)
    price_lower: Series[PriceUSDPerMWh] = Field(coerce=True)
    price_upper: Series[PriceUSDPerMWh] = Field(coerce=True)
    utilization: Series[Fraction] = Field(coerce=True)
    status: Series[str]


@dataclasses.dataclass(frozen=True, slots=True)
class Intervals:
    left: NDArray[np.float]
    right: NDArray[np.float]

    @classmethod
    def init(cls, x: NDArray[np.float], x0: np.float) -> Intervals:
        cx: NDArray[np.double] = np.cumsum(np.insert(x, 0, values=x0))
        return cls(left=cx[:-1], right=cx[1:])


@dataclasses.dataclass(frozen=True, slots=True)
class Clearance:
    load: PowerMW
    marginal_price: PriceUSDPerMWh
    stack: DataFrame[OfferStack]
    zone: ZoneId

    def plot(
        self,
        color_field: str | None = None,
        aggregate_load_color: str = "black",
        marginal_price_color: str = "red",
        price_axis_format: str = "$.0f",
        price_axis_title="marginal price",
        quantity_axis_title: str = "quantity [MW]",
    ) -> alt.LayerChart:
        """Produces plot of an offer stack."""

        stack: DataFrame[OfferStack] = self.stack
        load: PowerMW = self.load
        marginal_price: PriceUSDPerMWh = self.marginal_price

        offers_chart = alt.Chart(stack.reset_index())
        aggregate_chart = alt.Chart(
            pd.DataFrame(
                {
                    LOAD_KEY: load,
                    PRICE_LOCAL_KEY: marginal_price,
                },
                index=[0],
            )
        )
        return (
            offers_chart.mark_rect().encode(
                x=alt.X(OfferStack.capacity_left),
                x2=alt.X2(OfferStack.capacity_right),
                y=alt.Y(OfferStack.price_lower).scale(domainMin=0.0),
                y2=alt.Y2(OfferStack.price_upper),
                color=alt.Color(color_field or OfferStack.status),
                tooltip=[
                    OfferStack.generator,
                    OfferStack.tranche,
                    OfferStack.status,
                ],
            )
            + aggregate_chart.mark_rule(color=aggregate_load_color).encode(
                x=alt.X(LOAD_KEY, axis=alt.Axis(title=quantity_axis_title)),
                y=alt.Y().scale(domainMin=0.0),
                tooltip=[LOAD_KEY],
            )
            + aggregate_chart.mark_rule(color=marginal_price_color).encode(
                x=alt.X(),
                y=alt.Y(
                    PRICE_LOCAL_KEY,
                    axis=alt.Axis(format=price_axis_format, title=price_axis_title),
                ),
                tooltip=[PRICE_LOCAL_KEY],
            )
        )

    @classmethod
    def make_stacks(
        cls,
        generators: DataFrame[Generators],
        lines: DataFrame[Lines],
        lines_flow: DataFrame[LinesFlow],
        offers: DataFrame[Offers],
        offers_dispatched: DataFrame[OffersDispatched],
        zones: DataFrame[Zones],
        zones_price: DataFrame[ZonesPrice],
    ) -> dict[ZoneId, Clearance]:
        zone_from: Series[str] = lines[Lines.zone_from]
        zone_to: Series[str] = lines[Lines.zone_to]
        line_quantity: Series[PowerMW] = lines_flow[LinesFlow.quantity]
        negative_flow: Series[bool] = line_quantity < 0.0

        flow_zone_from: Series[str] = zone_from.mask(negative_flow, zone_to)
        flow_zone_to: Series[str] = zone_to.mask(negative_flow, zone_from)
        flow_quantity: Series[PowerMW] = line_quantity.abs()
        flow_price: Series[PriceUSDPerMWh] = pd.Series(
            zones_price.loc[flow_zone_from].values.flatten(),
            index=flow_zone_from.index,
            name=FLOW_PRICE,
        )

        offers_indexed = offers.set_index(OFFERS_INDEX_LABELS)
        offers_dispatched_indexed = offers_dispatched.set_index(OFFERS_INDEX_LABELS)
        offers_zones = pd.Series(
            generators[Generators.zone].loc[offers.generator].values,
            index=offers_indexed.index,
            name=Generators.zone,
        )
        generator_stack = pd.concat(
            [offers_indexed, offers_dispatched_indexed, offers_zones],
            axis="columns",
        )

        assert INFLOW_NAME not in generator_stack.index.unique(level=Offers.generator)
        inflow_stack = pd.DataFrame(
            {
                Offers.generator: pd.Series(INFLOW_NAME, index=lines.index),
                Offers.tranche: flow_zone_from,
                Offers.price: flow_price * +1,
                Offers.quantity: flow_quantity,
                OffersDispatched.quantity_dispatched: flow_quantity,
                Generators.zone: flow_zone_to,
            },
        )

        outflow_quantities: pd.DataFrame = (
            pd.concat([flow_zone_from, flow_quantity], axis="columns")
            .groupby(Lines.zone_from)
            .sum()
        )

        augmented_stack: pd.DataFrame = pd.concat(
            [
                generator_stack,
                inflow_stack.set_index(OFFERS_INDEX_LABELS),
            ],
            axis="rows",
        ).reset_index()

        def get_utilization(row: pd.Series) -> tuple[Fraction, str]:
            generator: GeneratorId = row[Offers.generator]
            quantity: PowerMW = row[Offers.quantity]
            dispatched: PowerMW = row[OffersDispatched.quantity_dispatched]
            utilization: Fraction = dispatched / quantity if quantity != 0.0 else 0.0
            # fmt: off
            description: str = (
                Utilization.IMPORTED if generator == INFLOW_NAME else
                Utilization.COMPLETE if  1.0 - BIND_TOL < utilization else
                Utilization.MARGINAL if 0.0 < utilization else
                Utilization.UNUSED
            ).value
            # fmt: on
            return utilization, description

        result: dict[ZoneId, cls] = {}

        for zone, stack_unsorted in augmented_stack.groupby(Generators.zone):
            stack = stack_unsorted.sort_values(Offers.price, ascending=True)
            quantity_exported: PowerMW = outflow_quantities[LinesFlow.quantity].get(
                zone, default=0.0
            )
            assert 0.0 <= quantity_exported

            capacity_pair = Intervals.init(
                stack[Offers.quantity].values, -quantity_exported
            )
            dispatched_pair = Intervals.init(
                stack[OffersDispatched.quantity_dispatched].values, -quantity_exported
            )
            temporary: pd.DataFrame = stack.apply(
                get_utilization, axis=1, result_type="expand"
            )

            stack[OfferStack.capacity_left] = capacity_pair.left
            stack[OfferStack.capacity_right] = capacity_pair.right
            stack[OfferStack.dispatched_left] = dispatched_pair.left
            stack[OfferStack.dispatched_right] = dispatched_pair.right
            stack[OfferStack.price_lower] = 0.0
            stack[OfferStack.price_upper] = stack[Offers.price]
            stack[OfferStack.utilization] = temporary[0]
            stack[OfferStack.status] = temporary[1]

            OfferStack.validate(stack)

            result[zone] = cls(
                load=zones.at[zone, Zones.load],
                marginal_price=zones_price.at[zone, ZonesPrice.price],
                stack=stack,
                zone=zone,
            )

        return result


def _main():
    from mini_iso.dashboard import LmpPricer, make_zone_stacks

    inputs: Input = load_system(constrained=True)
    pricer = LmpPricer.from_inputs(inputs)
    chart: alt.VConc = make_zone_stacks(pricer)
    file_path = pathlib.Path(__file__).with_suffix(".html")
    chart.save(file_path)
    print(f"Saved plot: {file_path}")


if __name__ == "__main__":
    _main()
