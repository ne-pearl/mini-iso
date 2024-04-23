from __future__ import annotations
import numpy as np
import pandas as pd
from pandera.typing import DataFrame, Series
import panel as pn
import param as pm
from mini_iso.clearance import Status, clear_auction
from mini_iso.typing import (
    OFFERS_INDEX_LABELS,
    GeneratorId,
    Generators,
    Input,
    Lines,
    LinesFlow,
    Offers,
    OffersDispatched,
    PaymentUSDPerH,
    PowerMW,
    PriceUSDPerMWh,
    Solution,
    ZoneId,
    Zones,
    ZonesPrice,
)
from mini_iso.miscellaneous import (
    INDICATOR_FONT_SIZES,
    tabulator_item,
    admittance_siemens,
    payment_usd_per_h,
    power_megawatts,
    price_usd_per_mwh,
)


def _validate_inputs(
    generators: DataFrame[Generators],
    lines: DataFrame[Lines],
    offers: DataFrame[Offers],
    zones: DataFrame[Zones],
):
    Generators.validate(generators)
    Lines.validate(lines)
    Offers.validate(offers)
    Zones.validate(zones)


def _validate_outputs(
    lines_flow: DataFrame[LinesFlow],
    offers_dispatched: DataFrame[OffersDispatched],
    zones_price: DataFrame[ZonesPrice],
):
    LinesFlow.validate(lines_flow)
    OffersDispatched.validate(offers_dispatched)
    ZonesPrice.validate(zones_price)


class LmpPricer(pn.viewable.Viewer):

    # Inputs
    generators = pm.DataFrame(label="Generators")
    lines = pm.DataFrame(label="Lines")
    offers = pm.DataFrame(label="Offers")
    zones = pm.DataFrame(label="Nodes")

    # Outputs
    lines_flow = pm.DataFrame(label="Line Flows")
    offers_dispatched = pm.DataFrame(label="Offer Dispatch")
    zones_price = pm.DataFrame(label="Zone Prices")
    status = pm.String(label="Status")

    objective = pm.Number(label="Objective Value")
    payment_from_loads = pm.Number(label="From Loads")
    payment_to_generators = pm.Number(label="To Generators")
    cost_of_congestion = pm.Number(label="Congestion Charges")

    def __init__(
        self,
        generators: DataFrame[Generators],
        lines: DataFrame[Lines],
        offers: DataFrame[Offers],
        zones: DataFrame[Zones],
    ):
        _validate_inputs(generators=generators, lines=lines, offers=offers, zones=zones)

        # Reset index to allow editing since panel doesn't
        # currently support editing of multi-index dataframes
        offers_flat = offers.reset_index()

        super().__init__(
            generators=generators,
            lines=lines,
            offers=offers_flat,
            zones=zones,
            lines_flow=pd.DataFrame(columns=[LinesFlow.quantity], index=lines.index),
            offers_dispatched=pd.DataFrame(
                columns=[OffersDispatched.quantity_dispatched], index=offers_flat.index
            ),
            zones_price=pd.DataFrame(columns=[ZonesPrice.price], index=zones.index),
        )

        self.param.watch(
            self.on_update,
            [
                self.param.generators.name,
                self.param.lines.name,
                self.param.offers.name,
                self.param.zones.name,
            ],
        )
        self.recompute()

    def on_update(self, event: pm.parameterized.Event) -> None:
        assert event.obj is self
        assert event.type == "changed"
        assert event.what == "value"
        self.recompute()

    def recompute(self) -> None:
        print("Attempting to clear market...")

        inputs = Input(
            generators=self.generators,
            lines=self.lines,
            offers=self.offers.set_index(OFFERS_INDEX_LABELS),
            zones=self.zones,
        )

        solution: Solution | None
        status: Status
        status, solution = clear_auction(inputs)

        # FIXME: Eliminate intermediate conversion?
        offers_index = self.offers.set_index(OFFERS_INDEX_LABELS).index

        lines_flow: pd.Series
        offers_dispatched: pd.Series
        zones_price: pd.Series

        if status is not Status.OPTIMAL:
            print(f"Failure: {status.value} [{status.name}]")
            lines_flow = Series[PowerMW](
                data=0.0,
                index=self.lines.index,
                name=LinesFlow.quantity,
            )
            offers_dispatched = Series[PowerMW](
                data=0.0,
                index=offers_index,
                name=OffersDispatched.quantity_dispatched,
            )
            zones_price = Series[PriceUSDPerMWh](
                data=0.0,
                index=self.zones.index,
                name=ZonesPrice.price,
            )
        else:
            assert solution is not None
            lines_flow = solution.lines[LinesFlow.quantity]
            offers_dispatched = solution.offers[OffersDispatched.quantity_dispatched]
            zones_price = solution.zones[ZonesPrice.price]

        zones_load: Series[PowerMW] = self.zones[Zones.load]
        generators_zone: Series[ZoneId] = self.generators[Generators.zone]
        offers_generator = Series[GeneratorId](
            data=self.offers[Offers.generator].values,
            index=offers_index,
        )

        offers_zone = Series[ZoneId](
            data=generators_zone.loc[offers_generator].values,
            index=offers_index,
        )
        offers_nodal_price = Series[PriceUSDPerMWh](
            data=zones_price[offers_zone].values,
            index=offers_index,
        )

        payment_from_loads: PaymentUSDPerH = (zones_price * zones_load).sum()
        payment_to_generators: PaymentUSDPerH = (
            offers_nodal_price * offers_dispatched
        ).sum()
        cost_of_congestion: PaymentUSDPerH = payment_from_loads - payment_to_generators

        # param.parameterized.update batches events associated with the update
        self.param.update(
            lines_flow=pd.DataFrame(lines_flow, index=self.lines.index),
            offers_dispatched=pd.DataFrame(
                # reset multi-index employed in solve routine
                offers_dispatched.reset_index(),
                index=self.offers.index,
            ),
            zones_price=pd.DataFrame(zones_price, index=self.zones.index),
            status=status.name,
            objective=np.nan if solution is None else solution.objective,
            payment_from_loads=np.nan if solution is None else payment_from_loads,
            payment_to_generators=np.nan if solution is None else payment_to_generators,
            cost_of_congestion=np.nan if solution is None else cost_of_congestion,
        )

        _validate_outputs(
            lines_flow=self.lines_flow,
            offers_dispatched=self.offers_dispatched.set_index(OFFERS_INDEX_LABELS),
            zones_price=self.zones_price,
        )

    @classmethod
    def from_inputs(cls, inputs: Input) -> LmpPricer:
        return cls(
            generators=inputs.generators,
            lines=inputs.lines,
            zones=inputs.zones,
            offers=inputs.offers,
        )

    def inputs_panel(self) -> pn.Column:
        return pn.Tabs(
            tabulator_item(
                self.param.generators,
                # Locked because capacity doesn't affect pricing directly:
                # Rather, capacity is conveyed via the offers.
                # FIXME: Does it make sense to make zone editable?
                disabled=True,
                formatters={Generators.capacity: power_megawatts.formatter},
                show_columns=[
                    Generators.name,
                    Generators.zone,
                    Generators.capacity,
                ],
                text_align={Generators.capacity: power_megawatts.align},
            ),
            tabulator_item(
                self.param.lines,
                disabled=False,
                formatters={
                    Lines.capacity: power_megawatts.formatter,
                    Lines.susceptance: admittance_siemens.formatter,
                },
                show_index=False,
                text_align={
                    Lines.capacity: power_megawatts.align,
                    Lines.susceptance: admittance_siemens.align,
                },
                titles={
                    # Abbreviations to save width
                    Lines.zone_from: "from",
                    Lines.zone_to: "to",
                },
            ),
            tabulator_item(
                self.param.zones,
                disabled=False,
                formatters={Zones.load: power_megawatts.formatter},
                show_columns=[
                    Zones.name,
                    Zones.load,
                ],
                text_align={Zones.load: power_megawatts.align},
            ),
            tabulator_item(
                self.param.offers,
                disabled=False,
                formatters={
                    Offers.quantity: power_megawatts.formatter,
                    Offers.price: price_usd_per_mwh.formatter,
                },
                show_columns=[
                    Offers.generator,
                    Offers.tranche,
                    Offers.price,
                    Offers.quantity,
                ],
                show_index=False,  # NB: after call to reset_index
                text_align={
                    Offers.quantity: power_megawatts.align,
                    Offers.price: price_usd_per_mwh.align,
                },
            ),
            active=2,  # nodes tab
        )

    def status_panel(self) -> pn.viewable.Viewable:
        return pn.Row(
            pn.widgets.StaticText.from_param(self.param.status),
            pn.indicators.Number.from_param(
                self.param.objective,
                disabled=True,
                format=f"{{value:.2f}}{payment_usd_per_h.formatter['symbol']}",
                **INDICATOR_FONT_SIZES,
            ),
            pn.indicators.Number.from_param(
                self.param.payment_from_loads,
                disabled=True,
                format=f"{{value:,.2f}}{payment_usd_per_h.formatter['symbol']}",
                **INDICATOR_FONT_SIZES,
            ),
            pn.indicators.Number.from_param(
                self.param.payment_to_generators,
                disabled=True,
                format=f"{{value:,.2f}}{payment_usd_per_h.formatter['symbol']}",
                **INDICATOR_FONT_SIZES,
            ),
            pn.indicators.Number.from_param(
                self.param.cost_of_congestion,
                disabled=True,
                format=f"{{value:,.2f}}{payment_usd_per_h.formatter['symbol']}",
                **INDICATOR_FONT_SIZES,
            ),
        )

    def outputs_panel(self) -> pn.Column:
        return pn.Column(
            self.status_panel(),
            pn.Tabs(
                tabulator_item(
                    self.param.lines_flow,
                    name=self.param.lines.label,
                    formatters={LinesFlow.quantity: power_megawatts.formatter},
                    text_align={LinesFlow.quantity: power_megawatts.align},
                ),
                tabulator_item(
                    self.param.offers_dispatched,
                    name=self.param.offers.label,
                    formatters={
                        OffersDispatched.quantity_dispatched: power_megawatts.formatter
                    },
                    text_align={
                        OffersDispatched.quantity_dispatched: power_megawatts.align
                    },
                ),
                tabulator_item(
                    self.param.zones_price,
                    name=self.param.zones.label,
                    formatters={ZonesPrice.price: price_usd_per_mwh.formatter},
                    text_align={ZonesPrice.price: price_usd_per_mwh.align},
                ),
            ),
        )

    def __panel__(self) -> pn.viewable.Viewable:
        return pn.template.VanillaTemplate(
            main=[
                pn.Row(
                    pn.Card(self.inputs_panel(), title="Inputs"),
                    pn.Card(self.outputs_panel(), title="Outputs"),
                )
            ],
            sidebar=[],
            sidebar_width=0,
            title="Mini-ISO: Back-End",
        )
