from typing import Final
import altair as alt
import numpy as np
from pandera.typing import DataFrame, Series
import pandas as pd
import panel as pn
import param as pm
from mini_iso.offer_stacks import OfferStack
from mini_iso.pricer import OFFERS_INDEX_LABELS, LmpPricer
from mini_iso.dataframes import (
    Fraction,
    GeneratorId,
    Generators,
    Lines,
    LinesFlow,
    Offers,
    OffersDispatched,
    PowerMW,
    SpatialCoordinate,
    ZoneId,
    Zones,
    ZonesPrice,
)
from mini_iso.clearance import (
    BIND_TOL,
    GeneratorsOutput,
    LinesOutput,
    OffersOutput,
    ZonesOutput,
)

TAB_GRAPHICAL: Final[str] = "Graphical"
TAB_TABULAR: Final[str] = "Tabular"


def make_bar_chart(
    dataframe: DataFrame,
    field_color: str,
    field_x: str,
    field_y: str,
) -> alt.Chart:
    return (
        alt.Chart(dataframe)
        .mark_bar()
        .encode(
            x=alt.X(field=field_x, type="nominal"),
            y=alt.Y(field=field_y, type="quantitative"),
            color=alt.Color(field=field_color, type="nominal"),
        )
    )


def _augment_generators_dataframe(pricer: LmpPricer, offers: DataFrame) -> DataFrame:

    generators_dispatched: Series[PowerMW] = offers.groupby(Offers.generator)[
        OffersDispatched.quantity_dispatched
    ].sum()
    assert generators_dispatched.index.name == Offers.generator
    generators_dispatched.index.name = GeneratorsOutput.name
    generators_capacity: Series[PowerMW] = pricer.generators[Generators.capacity]

    assert pricer.generators.index.name == GeneratorsOutput.name
    return DataFrame[GeneratorsOutput](
        {
            GeneratorsOutput.zone: pricer.generators[Generators.zone],
            GeneratorsOutput.capacity: generators_capacity,
            GeneratorsOutput.dispatched: generators_dispatched,
            GeneratorsOutput.utilization: (
                generators_dispatched / generators_capacity
            ).fillna(0.0),
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
            LinesOutput.abs_flow: lines_flow_abs,
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
    zone = pd.Series(zone_of_generators[generator_of_offers].values, index=offers.index)

    # Utilization of each offer
    quantity_offered: Series[PowerMW] = offers[Offers.quantity]
    quantity_dispatched: Series[PowerMW] = offers[OffersDispatched.quantity_dispatched]
    offers_utilization: Series[Fraction] = (
        quantity_dispatched / quantity_offered
    ).fillna(0.0)
    # fmt: off
    offers_is_marginal: Series[bool] = (
        (BIND_TOL <= offers_utilization) 
                  & (offers_utilization <= 1.0 - BIND_TOL)
    )

    assert all(offers_utilization >= 0.0)
    assert all(offers_utilization <= 1.0)

    # fmt: on
    return DataFrame[OffersOutput](
        {
            OffersOutput.price: offers[Offers.price].values,
            OffersOutput.quantity: quantity_offered.values,
            OffersOutput.quantity_dispatched: quantity_dispatched.values,
            OffersOutput.utilization: offers_utilization.values,
            OffersOutput.is_marginal: offers_is_marginal.values,
            OffersOutput.zone: zone.values,
        },
        index=pd.MultiIndex.from_arrays(
            [offers[Offers.generator], offers[Offers.tranche]],
            names=[Offers.generator, Offers.tranche],
        ),
    ).reset_index()


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
        # Combined offers data
        offers: DataFrame = pd.concat(
            (
                self.pricer.offers.set_index(OFFERS_INDEX_LABELS),
                self.pricer.offers_dispatched.set_index(OFFERS_INDEX_LABELS),
            ),
            axis="columns",
        ).reset_index()
        self.generators = _augment_generators_dataframe(self.pricer, offers=offers)
        self.lines = _augment_lines_dataframe(self.pricer)
        self.offers = _augment_offers_dataframe(self.pricer, offers=offers)
        self.zones = _augment_zones_dataframe(self.pricer, generators=self.generators)

    def generators_panel(self) -> pn.viewable.Viewable:
        field_select = pn.widgets.Select(
            options=[
                GeneratorsOutput.utilization,
                GeneratorsOutput.dispatched,
                GeneratorsOutput.capacity,
            ]
        )
        return pn.Tabs(
            (
                TAB_GRAPHICAL,
                pn.Column(
                    field_select,
                    pn.pane.Vega(
                        pn.bind(
                            make_bar_chart,
                            dataframe=self.param.generators,
                            field_color=GeneratorsOutput.zone,
                            field_x=GeneratorsOutput.name,
                            field_y=field_select,
                        )
                    ),
                ),
            ),
            (
                TAB_TABULAR,
                pn.widgets.Tabulator.from_param(
                    self.param.generators,
                    show_index=False,
                ),
            ),
        )

    def lines_panel(self) -> pn.viewable.Viewable:

        def make_network_plot(
            dataframe_lines: DataFrame,
            dataframe_nodes: DataFrame,
            color_field_lines: str,
            color_field_nodes: str,
        ) -> alt.Chart:

            circle_radius: float = 20.0

            # Default font size is 11
            # https://altair-viz.github.io/user_guide/marks/text.html
            font_size = alt.value(11)

            lines_plot = (
                alt.Chart(dataframe_lines)
                .mark_rule(color="black")
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
                        LinesOutput.name,
                        LinesOutput.slack,
                        LinesOutput.utilization,
                        LinesOutput.is_critical,
                        LinesOutput.abs_flow,
                        LinesOutput.capacity,
                    ],
                )
                .project(type="identity", reflectY=True)
                # .properties(height=400)
            )

            line_midpoints_plot = (
                alt.Chart(dataframe_lines)
                .mark_point()
                .encode(
                    x=alt.X(LinesOutput.x_mid),
                    y=alt.Y(LinesOutput.y_mid),
                    color=alt.Color(color_field_lines, legend=None),
                    size=alt.value(10),
                )
            )
            line_labels_plot = line_midpoints_plot.mark_text(
                align="center",
                baseline="middle",
                dx=10,
                dy=10,
            ).encode(
                color=alt.value("black"),
                size=font_size,
                text=alt.Text(color_field_lines),
            )

            nodes_plot = (
                alt.Chart(dataframe_nodes)
                .mark_circle()
                .project(type="identity", reflectY=True)
                .encode(
                    x=alt.X(ZonesOutput.x, type="quantitative"),
                    y=alt.Y(ZonesOutput.y, type="quantitative"),
                    color=alt.Color(color_field_nodes, legend=None),
                    size=alt.value(np.pi * circle_radius**2),  # actually area?
                    tooltip=[
                        ZonesOutput.name,
                        ZonesOutput.price,
                        ZonesOutput.capacity,
                        ZonesOutput.dispatched,
                    ],
                )
            )
            node_names_plot = nodes_plot.mark_text(
                align="center",
                baseline="middle",
                dx=0,
                dy=10,
            ).encode(
                color=alt.value("black"),
                size=font_size,
                text=alt.Text(ZonesOutput.name),
            )

            # node_labels_plot = nodes_plot.mark_text(
            #     align="center",
            #     baseline="middle",
            #     dx=0,
            #     dy=-10,
            # ).encode(
            #     color=alt.value("black"),
            #     size=font_size,
            #     text=alt.Text(color_field_nodes),
            # )

            return (
                alt.layer(
                    lines_plot,
                    line_midpoints_plot,
                    line_labels_plot,
                    nodes_plot,
                    node_names_plot,
                    # node_labels_plot,
                )
                .resolve_scale(color="independent")
                .configure_axis(disable=True, grid=False, domain=False)
                .properties(
                    height=600,
                    width=800,
                    # height="container",
                    # width="container",
                )
                .interactive()
            )

        lines_select = pn.widgets.Select(
            options=[
                LinesOutput.slack,
                LinesOutput.utilization,
                LinesOutput.is_critical,
                LinesOutput.abs_flow,
                LinesOutput.capacity,
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
            dataframe_lines=self.param.lines,
            dataframe_nodes=self.param.zones,
            color_field_lines=lines_select,
            color_field_nodes=nodes_select,
        )

        return pn.Tabs(
            (
                TAB_GRAPHICAL,
                pn.Column(
                    pn.Row(
                        lines_select,
                        nodes_select,
                    ),
                    pn.pane.Vega(network_chart, height=800),
                ),
            ),
            (
                TAB_TABULAR,
                pn.widgets.Tabulator.from_param(
                    self.param.lines,
                    show_index=False,
                ),
            ),
        )

    def offers_panel(self) -> pn.viewable.Viewable:

        offers = self.offers.set_index(OFFERS_INDEX_LABELS)
        zones = self.zones.set_index(Zones.name)
        generators = self.generators.set_index(Generators.name)

        # TODO: Refactor to tabulate marginal price for each zone
        offer_stacks: dict[str, OfferStack] = OfferStack.from_offers_by_zone(
            offers=offers,
            zones=zones,
            generators=generators,
        )
        offer_stacks_chart: alt.VConcatChart = (
            alt.vconcat(
                *(
                    zone_stack.plot().properties(title=zone)
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
        offer_stack_chart: alt.LayerChart = offer_stack.plot().interactive()

        return pn.Tabs(
            (
                TAB_GRAPHICAL,
                pn.Row(
                    pn.Card(
                        offer_stack_chart,
                        title="All Zones",
                    ),
                    pn.Card(
                        offer_stacks_chart,
                        title="Separate Zones",
                        collapsed=False,
                    ),
                ),
            ),
            (
                TAB_TABULAR,
                pn.widgets.Tabulator.from_param(
                    self.param.offers,
                    show_index=False,
                ),
            ),
        )

    def zones_panel(self) -> pn.viewable.Viewable:
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
                            make_bar_chart,
                            dataframe=self.param.zones,
                            field_color=ZonesOutput.name,
                            field_x=ZonesOutput.name,
                            field_y=field_select,
                        )
                    ),
                ),
            ),
            (
                TAB_TABULAR,
                pn.widgets.Tabulator.from_param(
                    self.param.zones,
                    show_index=False,
                ),
            ),
        )

    def __panel__(self) -> pn.viewable.Viewable:

        return pn.template.VanillaTemplate(
            main=[
                pn.Row(
                    pn.Card(self.pricer.inputs_panel(), title="Inputs"),
                    # pn.Card(self.lines_panel(), title="Lines"),
                    # pn.Card(self.generators_panel(), title="Generators"),
                    # pn.Card(self.offers_panel(), title="Offers"),
                    # pn.Card(self.zones_panel(), title="Zones"),
                    pn.Tabs(
                        ("Lines", self.lines_panel()),
                        ("Generators", self.generators_panel()),
                        ("Offers", self.offers_panel()),
                        ("Zones", self.zones_panel()),
                    ),
                )
            ],
            sidebar=[],
            sidebar_width=0,
            title="Mini-ISO: Dashboard",
        )
