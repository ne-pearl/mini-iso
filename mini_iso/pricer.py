from __future__ import annotations
import pandas as pd
from pandera.typing import DataFrame
import panel as pn
import param as pm
from mini_iso.clearance import Status, clear_auction
from mini_iso.typing import (
    OFFERS_INDEX_LABELS,
    Generators,
    Input,
    Lines,
    LinesFlow,
    Offers,
    OffersDispatched,
    Solution,
    Zones,
    ZonesPrice,
)
from mini_iso.panel_helpers import (
    tabulator_item,
    admittance_siemens,
    money_dollars,
    power_megawatts,
)
from mini_iso.panel_helpers import index_digits_key


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
    zones = pm.DataFrame(label="Zones")

    # Outputs
    lines_flow = pm.DataFrame(label="Line Flows")
    offers_dispatched = pm.DataFrame(label="Offer Dispatch")
    zones_price = pm.DataFrame(label="Zone Prices")

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

        lines_flow: pd.Series
        offers_dispatched: pd.Series
        zones_price: pd.Series

        if status is not Status.OPTIMAL:
            print(f"Failure: {status.value} [{status.name}]")
            lines_flow = pd.Series(
                data=0.0,
                index=self.lines.index,
                name=LinesFlow.quantity,
            )
            offers_dispatched = pd.Series(
                data=0.0,
                # FIXME: Eliminate intermediate conversion?
                index=self.offers.set_index(OFFERS_INDEX_LABELS).index,
                name=OffersDispatched.quantity_dispatched,
            )
            zones_price = pd.Series(
                data=0.0,
                index=self.zones.index,
                name=ZonesPrice.price,
            )
        else:
            assert solution is not None
            lines_flow = solution.lines[LinesFlow.quantity]
            offers_dispatched = solution.offers[
                OffersDispatched.quantity_dispatched
            ].sort_index(key=index_digits_key)
            zones_price = solution.zones[ZonesPrice.price]

        self.lines_flow = pd.DataFrame(lines_flow, index=self.lines.index)
        self.offers_dispatched = pd.DataFrame(
            # reset multi-index employed in solve routine
            offers_dispatched.reset_index(),
            index=self.offers.index,
        )
        self.zones_price = pd.DataFrame(zones_price, index=self.zones.index)

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
                show_columns=[
                    Lines.name,
                    Lines.zone_from,
                    Lines.zone_to,
                    Lines.capacity,
                    Lines.susceptance,
                ],
                text_align={
                    Lines.capacity: power_megawatts.align,
                    Lines.susceptance: admittance_siemens.align,
                },
            ),
            tabulator_item(
                self.param.offers,
                disabled=False,
                formatters={
                    Offers.quantity: power_megawatts.formatter,
                    Offers.price: money_dollars.formatter,
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
                    Offers.price: money_dollars.align,
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
        )

    def outputs_panel(self) -> pn.Column:
        return pn.Tabs(
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
                formatters={ZonesPrice.price: money_dollars.formatter},
                text_align={ZonesPrice.price: money_dollars.align},
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
