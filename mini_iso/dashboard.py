import collections
import functools
from typing import Final
import altair as alt
import numpy as np
from pandera.typing import DataFrame, Series
import pandas as pd
import panel as pn
import param as pm
from mini_iso.offer_stacks import Clearance
from mini_iso.offer_stacks_ideal import OfferStack
from mini_iso.miscellaneous import labeled
from mini_iso.pricer import LmpPricer
from mini_iso.typing import (
    OFFERS_INDEX_LABELS,
    Fraction,
    GeneratorId,
    Generators,
    GeneratorsOutput,
    Lines,
    LinesOutput,
    LinesFlow,
    Offers,
    OffersDispatched,
    OffersOutput,
    PaymentUSDPerH,
    PowerMW,
    PriceUSDPerMWh,
    SpatialCoordinate,
    ZoneId,
    Zones,
    ZonesPrice,
    ZonesOutput,
)
from mini_iso.miscellaneous import (
    BIND_TOL,
    admittance_siemens,
    boolean_check,
    fraction_percentage,
    payment_usd_per_h,
    price_usd_per_mwh,
    power_megawatts,
    real_unspecified,
    tristate_check,
)

TAB_GRAPHICAL: Final[str] = "Graphical"
TAB_TABULAR: Final[str] = "Tabular"


def make_bar_chart(
    dataframe: DataFrame,
    field_color: str,
    field_x: str,
    field_y: str,
    caller: str | None = None,
) -> alt.Chart:
    if caller:
        print(f"{caller} -> make_bar_chart")
    return (
        alt.Chart(dataframe)
        .mark_bar()
        .encode(
            x=alt.X(field=field_x, type="nominal"),
            y=alt.Y(field=field_y, type="quantitative"),
            color=alt.Color(field=field_color, type="nominal"),
        )
    )


def make_zone_stacks(pricer: LmpPricer) -> alt.VConcatChart:
    offer_stacks: dict[ZoneId, Clearance] = Clearance.make_stacks(
        generators=pricer.generators,
        lines=pricer.lines,
        lines_flow=pricer.lines_flow,
        offers=pricer.offers,
        offers_dispatched=pricer.offers_dispatched,
        zones=pricer.zones,
        zones_price=pricer.zones_price,
    )

    return alt.vconcat(
        *(
            zone_stack.plot().properties(title=zone)
            for zone, zone_stack in offer_stacks.items()
        )
    ).interactive()


def _augment_generators_dataframe(pricer: LmpPricer, offers: DataFrame) -> DataFrame:
    generators_dispatched: Series[PowerMW] = offers.groupby(Offers.generator)[
        OffersDispatched.quantity_dispatched
    ].sum()
    assert generators_dispatched.index.name == Offers.generator
    generators_dispatched.index.name = GeneratorsOutput.name
    generators_capacity: Series[PowerMW] = pricer.generators[Generators.capacity]

    x: Series[SpatialCoordinate] = pricer.generators[Generators.x]
    y: Series[SpatialCoordinate] = pricer.generators[Generators.y]

    zone: Series[ZoneId] = pricer.generators[Generators.zone]
    x_zone = Series[SpatialCoordinate](
        data=pricer.zones[Zones.x][zone].values,
        index=pricer.generators.index,
    )
    y_zone = Series[SpatialCoordinate](
        data=pricer.zones[Zones.y][zone].values,
        index=pricer.generators.index,
    )

    nodal_prices = Series[PriceUSDPerMWh](
        data=pricer.zones_price[ZonesPrice.price].loc[zone].values,
        index=pricer.generators.index,
    )
    revenues: Series[PaymentUSDPerH] = nodal_prices * generators_dispatched

    assert pricer.generators.index.name == GeneratorsOutput.name
    return DataFrame[GeneratorsOutput](
        {
            GeneratorsOutput.zone: pricer.generators[Generators.zone],
            GeneratorsOutput.capacity: generators_capacity,
            GeneratorsOutput.dispatched: generators_dispatched,
            GeneratorsOutput.utilization: (
                generators_dispatched / generators_capacity
            ).fillna(0.0),
            GeneratorsOutput.x: x,
            GeneratorsOutput.y: y,
            GeneratorsOutput.x_zone: x_zone,
            GeneratorsOutput.y_zone: y_zone,
            GeneratorsOutput.x_mid: (x + x_zone) * 0.5,
            GeneratorsOutput.y_mid: (y + y_zone) * 0.5,
            GeneratorsOutput.nodal_price: nodal_prices,
            GeneratorsOutput.revenue: revenues,
        },
        index=pricer.generators.index,
    ).reset_index()


def _augment_lines_dataframe(pricer: LmpPricer) -> DataFrame:
    x_of_zones: Series[SpatialCoordinate] = pricer.zones[Zones.x]
    y_of_zones: Series[SpatialCoordinate] = pricer.zones[Zones.y]
    zone_from_of_lines: Series[ZoneId] = pricer.lines[Lines.zone_from]
    zone_to_of_lines: Series[ZoneId] = pricer.lines[Lines.zone_to]
    lines_flow: Series[PowerMW] = pricer.lines_flow[LinesFlow.quantity]
    lines_flow_abs: Series[PowerMW] = lines_flow.abs()
    lines_capacity: Series[PowerMW] = pricer.lines[Lines.capacity]
    lines_slack: Series[PowerMW] = lines_capacity - lines_flow_abs
    lines_utilization: Series[Fraction] = 1.0 - (lines_slack / lines_capacity).fillna(
        0.0
    )
    lines_is_critical: Series[bool] = (1.0 - BIND_TOL) <= lines_utilization

    x_from = x_of_zones[zone_from_of_lines].values
    y_from = y_of_zones[zone_from_of_lines].values
    x_to = x_of_zones[zone_to_of_lines].values
    y_to = y_of_zones[zone_to_of_lines].values

    return DataFrame[LinesOutput](
        {
            LinesOutput.zone_from: pricer.lines[Lines.zone_from],
            LinesOutput.zone_to: pricer.lines[Lines.zone_to],
            LinesOutput.susceptance: pricer.lines[Lines.susceptance],
            LinesOutput.quantity: lines_flow,
            LinesOutput.quantity_abs: lines_flow_abs,
            LinesOutput.capacity: lines_capacity,
            LinesOutput.slack: lines_slack,
            LinesOutput.utilization: lines_utilization,
            LinesOutput.is_critical: lines_is_critical,
            LinesOutput.x_from: x_from,
            LinesOutput.y_from: y_from,
            LinesOutput.x_to: x_to,
            LinesOutput.y_to: y_to,
            LinesOutput.x_mid: (x_from + x_to) * 0.5,
            LinesOutput.y_mid: (y_from + y_to) * 0.5,
        },
        index=pricer.lines.index,
    ).reset_index()


def _augment_offers_dataframe(pricer: LmpPricer, offers: DataFrame) -> DataFrame:
    generator_of_offers: Series[GeneratorId] = offers[Offers.generator]
    zone_of_generators: Series[ZoneId] = pricer.generators[Generators.zone]
    zone = Series[ZoneId](
        data=zone_of_generators[generator_of_offers].values,
        index=offers.index,
        name=OffersOutput.zone,
    )

    # Utilization of each offer
    quantity_offered: Series[PowerMW] = offers[Offers.quantity]
    quantity_dispatched: Series[PowerMW] = offers[OffersDispatched.quantity_dispatched]
    offers_utilization = Series[Fraction](
        quantity_dispatched / quantity_offered,
        name=OffersOutput.utilization,
    ).fillna(0.0)

    nodal_prices = Series[PriceUSDPerMWh](
        data=pricer.zones_price[ZonesPrice.price].loc[zone].values,
        index=offers.index,
        name=OffersOutput.nodal_price,
    )
    revenues = Series[PaymentUSDPerH](
        nodal_prices * quantity_dispatched,
        name=OffersOutput.revenue,
    )

    def is_marginal(utilization: Fraction) -> bool | None:
        if utilization == 0.0:
            return False
        if 0.0 < utilization < 1.0 - BIND_TOL:
            return True
        # We want to return None here, but Pandera doesn't appear to
        # support the combination:
        #   is_marginal: Series[bool] = Field(nullable=True)
        # so we return False instead
        return False

    offers_is_marginal = Series[bool | None](
        offers_utilization.map(is_marginal),
        name=OffersOutput.is_marginal,
    )

    result = DataFrame[OffersOutput](
        {
            OffersOutput.price: offers[Offers.price].values,
            OffersOutput.quantity: quantity_offered.values,
            OffersOutput.quantity_dispatched: quantity_dispatched.values,
            OffersOutput.utilization: offers_utilization.values,
            OffersOutput.is_marginal: offers_is_marginal.values,
            OffersOutput.zone: zone.values,
            OffersOutput.nodal_price: nodal_prices.values,
            OffersOutput.revenue: revenues.values,
        },
        index=pd.MultiIndex.from_arrays(
            [offers[Offers.generator], offers[Offers.tranche]],
            names=OFFERS_INDEX_LABELS,
        ),
    )

    result2 = DataFrame[OffersOutput](
        pd.concat(
            (
                offers[Offers.generator],
                offers[Offers.tranche],
                offers[Offers.price],
                quantity_offered,
                quantity_dispatched,
                offers_utilization,
                offers_is_marginal,
                zone,
                nodal_prices,
                revenues,
            ),
            axis="columns",
        ).set_index(OFFERS_INDEX_LABELS)
    )

    assert result.equals(result2)

    return result.reset_index()


def _augment_zones_dataframe(
    pricer: LmpPricer,
    generators: DataFrame[GeneratorsOutput],
) -> DataFrame:
    # Sum dispatched output over each zone's generators:
    zones_generation_temporary: Series[PowerMW] = generators.groupby(
        GeneratorsOutput.zone
    )[
        [
            GeneratorsOutput.capacity,
            GeneratorsOutput.dispatched,
        ]
    ].sum()

    # Zones needn't all have generators: Update index and replace NaN with 0.0
    zones_generation: pd.DataFrame = pd.merge(
        left=zones_generation_temporary,
        left_index=True,
        right=pricer.zones[[]],  # index only
        right_index=True,
        how="right",
    ).fillna(0.0)

    zones_dispatched: Series[PowerMW] = zones_generation[GeneratorsOutput.dispatched]
    zones_capacity: Series[PowerMW] = zones_generation[GeneratorsOutput.capacity]
    # Replace 0/0=NaN with 0
    zones_utilization: Series[Fraction] = (zones_dispatched / zones_capacity).fillna(
        0.0
    )

    return DataFrame[ZonesOutput](
        {
            ZonesOutput.price: pricer.zones_price[ZonesPrice.price],
            ZonesOutput.load: pricer.zones[Zones.load],
            ZonesOutput.dispatched: zones_dispatched,
            ZonesOutput.capacity: zones_capacity,
            ZonesOutput.utilization: zones_utilization,
            ZonesOutput.x: pricer.zones[Zones.x],
            ZonesOutput.y: pricer.zones[Zones.y],
        },
        index=pricer.zones.index,
    ).reset_index()


class LmpDashboard(pm.Parameterized):
    pricer = pm.ClassSelector(class_=LmpPricer, allow_refs=True, instantiate=False)

    generators = pm.DataFrame()
    lines = pm.DataFrame()
    offers = pm.DataFrame()
    zones = pm.DataFrame()

    @pn.depends("pricer.param", on_init=True, watch=True)
    def _refresh(self) -> None:
        print("LmpDashboard._refresh")
        # Combined offers data
        offers_new: DataFrame = pd.concat(
            (
                self.pricer.offers.set_index(OFFERS_INDEX_LABELS),
                self.pricer.offers_dispatched.set_index(OFFERS_INDEX_LABELS),
            ),
            axis="columns",
        ).reset_index()

        generators_new = _augment_generators_dataframe(self.pricer, offers=offers_new)
        lines_new = _augment_lines_dataframe(self.pricer)
        offers_new = _augment_offers_dataframe(self.pricer, offers=offers_new)
        zones_new = _augment_zones_dataframe(self.pricer, generators=generators_new)

        self.param.update(
            generators=generators_new,
            lines=lines_new,
            offers=offers_new,
            zones=zones_new,
        )

    def generators_panel(self) -> pn.viewable.Viewable:
        print("  LmpDashboard.generators_panel...")
        field_select = pn.widgets.Select(
            options=[
                GeneratorsOutput.dispatched,
                GeneratorsOutput.capacity,
                GeneratorsOutput.utilization,
            ]
        )
        return pn.Tabs(
            (
                TAB_GRAPHICAL,
                pn.Column(
                    field_select,
                    pn.pane.Vega(
                        pn.bind(
                            functools.partial(
                                make_bar_chart,
                                caller="    LmpDashboard.generators_panel",
                            ),
                            dataframe=self.param.generators,
                            field_color=GeneratorsOutput.zone,
                            field_x=GeneratorsOutput.name,
                            field_y=field_select,
                            watch=True,
                        )
                    ),
                ),
            ),
            (
                TAB_TABULAR,
                pn.widgets.Tabulator.from_param(
                    self.param.generators,
                    formatters={
                        GeneratorsOutput.capacity: power_megawatts.formatter,
                        GeneratorsOutput.dispatched: power_megawatts.formatter,
                        GeneratorsOutput.utilization: fraction_percentage.formatter,
                        GeneratorsOutput.nodal_price: price_usd_per_mwh.formatter,
                        GeneratorsOutput.revenue: payment_usd_per_h.formatter,
                    },
                    hidden_columns=[
                        GeneratorsOutput.x,
                        GeneratorsOutput.y,
                        GeneratorsOutput.x_zone,
                        GeneratorsOutput.y_zone,
                        GeneratorsOutput.x_mid,
                        GeneratorsOutput.y_mid,
                    ],
                    show_index=False,
                    text_align={
                        GeneratorsOutput.capacity: power_megawatts.align,
                        GeneratorsOutput.dispatched: power_megawatts.align,
                        GeneratorsOutput.utilization: fraction_percentage.align,
                        GeneratorsOutput.nodal_price: price_usd_per_mwh.align,
                        GeneratorsOutput.revenue: payment_usd_per_h.align,
                    },
                    titles={
                        GeneratorsOutput.nodal_price: "LMP",
                    },
                ),
            ),
        )

    def network_panel(self) -> pn.viewable.Viewable:
        print("  LmpDashboard.network_panel")

        def make_network_plot(
            dataframe_generators: DataFrame,
            dataframe_lines: DataFrame,
            dataframe_zones: DataFrame,
            color_field_generators: str,
            color_field_lines: str,
            color_field_zones: str,
        ) -> alt.Chart:
            # Default font size is 11
            # https://altair-viz.github.io/user_guide/marks/text.html
            font_size: Final[dict] = alt.value(14)
            dy: Final[int] = 8
            circle_radius: Final[float] = 30.0
            format_strings: Final[dict[str, str]] = collections.defaultdict(
                # default format: "Currency"
                lambda: ".0f",
                # Specific overrides
                {
                    GeneratorsOutput.utilization: ".0%",
                    LinesOutput.is_critical: ".0",
                    LinesOutput.utilization: ".0%",
                    ZonesOutput.utilization: ".0%",
                },
            )

            generators_chart = alt.Chart(dataframe_generators)
            lines_chart = alt.Chart(dataframe_lines)
            zones_chart = alt.Chart(dataframe_zones)

            zones_line_plot = (
                lines_chart.mark_rule(color="black")
                .encode(
                    x=alt.X(
                        LinesOutput.x_from,
                        axis=alt.Axis(title="x"),
                        type="quantitative",
                    ),
                    y=alt.Y(
                        LinesOutput.y_from,
                        axis=alt.Axis(title="y"),
                        type="quantitative",
                    ),
                    x2=alt.X2(LinesOutput.x_to),
                    y2=alt.Y2(LinesOutput.y_to),
                    color=alt.Color(color_field_lines, legend=None),
                    size=alt.value(5),
                    tooltip=[
                        alt.Tooltip(LinesOutput.name),
                        alt.Tooltip(LinesOutput.zone_from),
                        alt.Tooltip(LinesOutput.zone_to),
                        alt.Tooltip(LinesOutput.slack, format=".2f"),
                        alt.Tooltip(LinesOutput.utilization, format=".0%"),
                        alt.Tooltip(LinesOutput.is_critical),
                        alt.Tooltip(LinesOutput.quantity, format=".0f"),
                        alt.Tooltip(LinesOutput.capacity, format=".0f"),
                    ],
                )
                .project(type="identity", reflectY=True)
                # .properties(height=400)
            )

            zones_line_midpoint_plot = lines_chart.mark_point().encode(
                x=alt.X(LinesOutput.x_mid),
                y=alt.Y(LinesOutput.y_mid),
                color=alt.Color(color_field_lines, legend=None),
                size=alt.value(10),
            )

            zones_line_label_plot = zones_line_midpoint_plot.mark_text(
                align="center",
                baseline="middle",
                dx=0,
                dy=0,
            ).encode(
                color=alt.value("black"),
                size=font_size,
                text=alt.Text(
                    color_field_lines,
                    format=format_strings[color_field_lines],
                    formatType="number",
                ),
            )

            zones_plot = (
                zones_chart.mark_square()
                .project(type="identity", reflectY=True)
                .encode(
                    x=alt.X(ZonesOutput.x, type="quantitative"),
                    y=alt.Y(ZonesOutput.y, type="quantitative"),
                    color=alt.Color(color_field_zones, legend=None),
                    size=alt.value(np.pi * circle_radius**2),  # actually area?
                    tooltip=[
                        ZonesOutput.name,
                        alt.Tooltip(ZonesOutput.price, format="$.2f"),
                        alt.Tooltip(ZonesOutput.load, format=".0f"),
                        alt.Tooltip(ZonesOutput.capacity, format=".0f"),
                        alt.Tooltip(ZonesOutput.dispatched, format=".0f"),
                        alt.Tooltip(ZonesOutput.utilization, format=".0%"),
                    ],
                )
            )

            zones_name_plot = zones_plot.mark_text(
                align="center",
                baseline="middle",
                dx=0,
                dy=-dy,  # negative means "upwards"
            ).encode(
                color=alt.value("black"),
                size=font_size,
                text=alt.Text(ZonesOutput.name),
            )

            zones_data_plot = zones_plot.mark_text(
                align="center",
                baseline="middle",
                dx=0,
                dy=+dy,  # positive means "downwards"
            ).encode(
                color=alt.value("black"),
                size=font_size,
                text=alt.Text(
                    color_field_zones,
                    format=format_strings[color_field_zones],
                    formatType="number",
                ),
            )

            generators_line_plot = (
                generators_chart.mark_rule(color="black")
                .encode(
                    x=alt.X(
                        GeneratorsOutput.x,
                        axis=alt.Axis(title="x"),
                        type="quantitative",
                    ),
                    y=alt.Y(
                        GeneratorsOutput.y,
                        axis=alt.Axis(title="y"),
                        type="quantitative",
                    ),
                    x2=alt.X2(GeneratorsOutput.x_zone),
                    y2=alt.Y2(GeneratorsOutput.y_zone),
                    color=alt.Color(color_field_generators, legend=None),
                    size=alt.value(5),
                    tooltip=[
                        alt.Tooltip(GeneratorsOutput.name),
                        alt.Tooltip(GeneratorsOutput.zone),
                        alt.Tooltip(GeneratorsOutput.dispatched, format=".0f"),
                        alt.Tooltip(GeneratorsOutput.capacity, format=".0f"),
                        alt.Tooltip(GeneratorsOutput.utilization, format=".0%"),
                    ],
                )
                .project(type="identity", reflectY=True)
                # .properties(height=400)
            )

            generators_line_midpoint_plot = generators_chart.mark_point().encode(
                x=alt.X(GeneratorsOutput.x_mid),
                y=alt.Y(GeneratorsOutput.y_mid),
                color=alt.Color(color_field_generators, legend=None),
                size=alt.value(10),
            )

            generators_line_label_plot = generators_line_midpoint_plot.mark_text(
                align="center",
                baseline="middle",
                dx=0,
                dy=0,
            ).encode(
                color=alt.value("black"),
                size=font_size,
                text=alt.Text(
                    color_field_generators,
                    format=format_strings[color_field_generators],
                    formatType="number",
                ),
            )

            generators_plot = (
                generators_chart.mark_circle()
                .project(type="identity", reflectY=True)
                .encode(
                    x=alt.X(GeneratorsOutput.x, type="quantitative"),
                    y=alt.Y(GeneratorsOutput.y, type="quantitative"),
                    color=alt.Color(color_field_generators, legend=None),
                    size=alt.value(np.pi * circle_radius**2),  # actually area?
                    tooltip=[
                        GeneratorsOutput.name,
                        alt.Tooltip(GeneratorsOutput.capacity, format=".0f"),
                        alt.Tooltip(GeneratorsOutput.dispatched, format=".0f"),
                        alt.Tooltip(GeneratorsOutput.utilization, format=".0%"),
                    ],
                )
            )

            generators_name_plot = generators_plot.mark_text(
                align="center",
                baseline="middle",
                dx=0,
                dy=0,
            ).encode(
                color=alt.value("black"),
                size=font_size,
                text=alt.Text(GeneratorsOutput.name),
            )

            return (
                alt.layer(
                    # Lower layers: Lines
                    zones_line_plot,
                    generators_line_plot,
                    # Middle layers: Nodes
                    zones_plot,
                    generators_plot,
                    # Upper layers: Text
                    generators_line_label_plot,
                    generators_line_midpoint_plot,
                    generators_name_plot,
                    zones_line_label_plot,
                    zones_line_midpoint_plot,
                    zones_name_plot,
                    zones_data_plot,
                )
                .resolve_scale(color="independent")
                .configure_axis(disable=True, grid=False, domain=False)
                .properties(
                    height="container",
                    width="container",
                )
                .interactive()
            )

        generators_select = pn.widgets.Select(
            options=[
                GeneratorsOutput.dispatched,
                GeneratorsOutput.capacity,
                GeneratorsOutput.utilization,
            ],
            name="Generators",
        )

        lines_select = pn.widgets.Select(
            options=[
                LinesOutput.quantity,
                LinesOutput.quantity_abs,
                LinesOutput.capacity,
                LinesOutput.utilization,
                LinesOutput.is_critical,
                LinesOutput.slack,
            ],
            name="Lines",
        )

        nodes_select = pn.widgets.Select(
            options=[
                ZonesOutput.price,
                ZonesOutput.load,
                ZonesOutput.capacity,
                ZonesOutput.dispatched,
                ZonesOutput.utilization,
            ],
            name="Zones",
        )

        network_chart: alt.Chart = pn.bind(
            make_network_plot,
            dataframe_generators=self.param.generators,
            dataframe_lines=self.param.lines,
            dataframe_zones=self.param.zones,
            color_field_generators=generators_select,
            color_field_lines=lines_select,
            color_field_zones=nodes_select,
            watch=True,
        )

        return pn.Tabs(
            (
                TAB_GRAPHICAL,
                pn.Column(
                    pn.Row(
                        generators_select,
                        lines_select,
                        nodes_select,
                    ),
                    pn.pane.Vega(network_chart, sizing_mode="stretch_height"),
                ),
            ),
            (
                TAB_TABULAR,
                pn.widgets.Tabulator.from_param(
                    self.param.lines,
                    formatters={
                        LinesOutput.susceptance: admittance_siemens.formatter,
                        LinesOutput.quantity: power_megawatts.formatter,
                        LinesOutput.quantity_abs: power_megawatts.formatter,
                        LinesOutput.capacity: power_megawatts.formatter,
                        LinesOutput.slack: power_megawatts.formatter,
                        LinesOutput.utilization: fraction_percentage.formatter,
                        LinesOutput.is_critical: boolean_check.formatter,
                        LinesOutput.x_from: real_unspecified.formatter,
                        LinesOutput.y_from: real_unspecified.formatter,
                        LinesOutput.x_to: real_unspecified.formatter,
                        LinesOutput.y_to: real_unspecified.formatter,
                        LinesOutput.x_mid: real_unspecified.formatter,
                        LinesOutput.y_mid: real_unspecified.formatter,
                    },
                    hidden_columns=[
                        LinesOutput.name,
                        LinesOutput.susceptance,
                        LinesOutput.x_from,
                        LinesOutput.y_from,
                        LinesOutput.x_to,
                        LinesOutput.y_to,
                        LinesOutput.x_mid,
                        LinesOutput.y_mid,
                    ],
                    show_index=False,
                    text_align={
                        LinesOutput.susceptance: admittance_siemens.align,
                        LinesOutput.quantity: power_megawatts.align,
                        LinesOutput.quantity_abs: power_megawatts.align,
                        LinesOutput.capacity: power_megawatts.align,
                        LinesOutput.slack: power_megawatts.align,
                        LinesOutput.utilization: fraction_percentage.align,
                        LinesOutput.is_critical: boolean_check.align,
                        LinesOutput.x_from: real_unspecified.align,
                        LinesOutput.y_from: real_unspecified.align,
                        LinesOutput.x_to: real_unspecified.align,
                        LinesOutput.y_to: real_unspecified.align,
                        LinesOutput.x_mid: real_unspecified.align,
                        LinesOutput.y_mid: real_unspecified.align,
                    },
                    titles={
                        LinesOutput.zone_from: "from",
                        LinesOutput.zone_to: "to",
                        LinesOutput.is_critical: "congested",
                        LinesOutput.quantity: "flow",
                        LinesOutput.quantity_abs: "|flow|",
                    },
                ),
            ),
        )

    def offers_panel(self) -> pn.viewable.Viewable:
        print("  LmpDashboard.offers_panel")

        def _refresh(offers_in: DataFrame, zones_in: DataFrame, pricer: LmpPricer):
            offers: DataFrame[OffersOutput] = offers_in.set_index(OFFERS_INDEX_LABELS)
            zones: DataFrame[ZonesOutput] = zones_in.set_index(Zones.name)

            offer_stacks_chart_zonal: alt.VConcatChart = make_zone_stacks(self.pricer)

            # TODO: Refactor to tabulate marginal price for each zone
            offer_stacks: dict[str, OfferStack] = OfferStack.from_offers_by_zone(
                offers=offers,
                zones=zones,
            )
            offer_stacks_chart_ideal: alt.VConcatChart = (
                alt.vconcat(
                    *(
                        zone_stack.plot(color_field=OffersOutput.generator).properties(
                            title=zone
                        )
                        for zone, zone_stack in offer_stacks.items()
                    )
                )
                .resolve_scale(color="independent")
                .interactive()
            )

            # TODO: Refactor to store marginal price
            offer_stack: OfferStack = OfferStack.from_offers(
                offers=offers,
                load=zones[ZonesOutput.load].sum(),
            )
            offer_stack_chart: alt.LayerChart = offer_stack.plot(
                color_field=OffersOutput.zone
            ).interactive()

            return pn.Tabs(
                (
                    "By Zone",
                    pn.Row(
                        labeled(pn.pane.Vega(offer_stacks_chart_zonal), label="Actual"),
                        labeled(
                            pn.pane.Vega(offer_stacks_chart_ideal), label="Isolated"
                        ),
                    ),
                ),
                (
                    "Ideal Aggregate",
                    pn.pane.Vega(offer_stack_chart, sizing_mode="stretch_height"),
                ),
            )

        return pn.Tabs(
            (
                TAB_GRAPHICAL,
                pn.bind(
                    _refresh,
                    offers_in=self.param.offers,
                    zones_in=self.param.zones,
                    pricer=self.param.pricer,
                    # watch=True,
                ),
            ),
            (
                TAB_TABULAR,
                pn.widgets.Tabulator.from_param(
                    self.param.offers,
                    formatters={
                        OffersOutput.price: price_usd_per_mwh.formatter,
                        OffersOutput.quantity: power_megawatts.formatter,
                        OffersOutput.quantity_dispatched: power_megawatts.formatter,
                        OffersOutput.utilization: fraction_percentage.formatter,
                        OffersOutput.is_marginal: tristate_check.formatter,
                        OffersOutput.nodal_price: price_usd_per_mwh.formatter,
                        OffersOutput.revenue: payment_usd_per_h.formatter,
                    },
                    hidden_columns=[
                        OffersOutput.zone,
                    ],
                    show_index=False,
                    text_align={
                        OffersOutput.price: price_usd_per_mwh.align,
                        OffersOutput.quantity: power_megawatts.align,
                        OffersOutput.quantity_dispatched: power_megawatts.align,
                        OffersOutput.utilization: fraction_percentage.align,
                        OffersOutput.is_marginal: tristate_check.align,
                        OffersOutput.nodal_price: price_usd_per_mwh.align,
                        OffersOutput.revenue: payment_usd_per_h.align,
                    },
                    titles={
                        OffersOutput.generator: "gen",
                        OffersOutput.tranche: "tr",
                        OffersOutput.price: "offer",
                        OffersOutput.quantity: "offer",
                        OffersOutput.quantity_dispatched: "dispatched",
                        OffersOutput.is_marginal: "marginal",
                        OffersOutput.nodal_price: "LMP",
                    },
                ),
            ),
        )

    def zones_panel(self) -> pn.viewable.Viewable:
        print("  LmpDashboard.zones_panel")
        field_select = pn.widgets.Select(
            options=[
                ZonesOutput.price,
                ZonesOutput.utilization,
                ZonesOutput.dispatched,
                ZonesOutput.capacity,
                ZonesOutput.load,
            ]
        )
        return pn.Tabs(
            (
                TAB_GRAPHICAL,
                pn.Column(
                    field_select,
                    pn.pane.Vega(
                        pn.bind(
                            functools.partial(
                                make_bar_chart,
                                caller="    LmpDashboard.zones_panel",
                            ),
                            dataframe=self.param.zones,
                            field_color=ZonesOutput.name,
                            field_x=ZonesOutput.name,
                            field_y=field_select,
                            watch=True,
                        )
                    ),
                ),
            ),
            (
                TAB_TABULAR,
                pn.widgets.Tabulator.from_param(
                    self.param.zones,
                    formatters={
                        ZonesOutput.price: price_usd_per_mwh.formatter,
                        ZonesOutput.load: power_megawatts.formatter,
                        ZonesOutput.dispatched: power_megawatts.formatter,
                        ZonesOutput.capacity: power_megawatts.formatter,
                        ZonesOutput.utilization: fraction_percentage.formatter,
                        ZonesOutput.x: real_unspecified.formatter,
                        ZonesOutput.y: real_unspecified.formatter,
                    },
                    hidden_columns=[
                        ZonesOutput.x,
                        ZonesOutput.y,
                    ],
                    show_index=False,
                    text_align={
                        ZonesOutput.price: price_usd_per_mwh.align,
                        ZonesOutput.load: power_megawatts.align,
                        ZonesOutput.dispatched: power_megawatts.align,
                        ZonesOutput.capacity: power_megawatts.align,
                        ZonesOutput.utilization: fraction_percentage.align,
                        ZonesOutput.x: real_unspecified.align,
                        ZonesOutput.y: real_unspecified.align,
                    },
                ),
            ),
        )

    def __panel__(self) -> pn.viewable.Viewable:
        print("LmpDashboard.__panel__")
        return pn.template.VanillaTemplate(
            main=[
                labeled(
                    pn.Tabs(
                        ("Lines", self.network_panel()),
                        ("Generators", self.generators_panel()),
                        ("Offers", self.offers_panel()),
                        ("Zones", self.zones_panel()),
                    ),
                    label="Outputs",
                ),
            ],
            sidebar=[
                labeled(self.pricer.inputs_panel(), label="Inputs"),
            ],
            sidebar_width=450,
            title="Mini-ISO: Dashboard",
        )
