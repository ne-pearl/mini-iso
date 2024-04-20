from __future__ import annotations
import panel as pn
import param as pm
from mini_iso.typing import (
    Offers,
    LinesFlow,
    OffersDispatched,
    ZonesPrice,
)
from mini_iso.miscellaneous import (
    price_usd_per_mwh,
    power_megawatts,
    tabulator_item,
)
from mini_iso.pricer import LmpPricer


class Auction(pn.viewable.Viewer):
    pricer = pm.ClassSelector(class_=LmpPricer, instantiate=False, label="LMP Pricer")
    offers_pending = pm.DataFrame(label="Pending Offers")

    submit = pm.Event(label="Submit")
    reset = pm.Event(label="Reset")

    lines_flow = pm.DataFrame(label="Line Flows", allow_refs=True, instantiate=False)
    offers_committed = pm.DataFrame(
        label="Offers Committed", allow_refs=True, instantiate=False
    )
    offers_dispatched = pm.DataFrame(
        label="Dispatched", allow_refs=True, instantiate=False
    )
    zones_price = pm.DataFrame(label="Prices", allow_refs=True, instantiate=False)

    def __init__(self, pricer: LmpPricer, **params):
        super().__init__(
            **params,
            pricer=pricer,
            offers_pending=pricer.offers.copy(),
            # Direct links
            lines_flow=pricer.param.lines_flow,
            offers_committed=pricer.param.offers,
            offers_dispatched=pricer.param.offers_dispatched,
            zones_price=pricer.param.zones_price,
        )

    @pn.depends("submit", watch=True)
    def _on_submit(self) -> None:
        print("Submitting offers to pricer...")
        self.pricer.offers = self.offers_pending.copy()

    @pn.depends("reset", watch=True)
    def _on_reset(self) -> None:
        self.offers_pending = self.pricer.offers.copy()

    def __panel__(self) -> pn.viewable.Viewable:
        assert all(self.lines_flow == self.pricer.lines_flow)
        assert all(self.offers_committed == self.pricer.offers)
        assert all(self.offers_dispatched == self.pricer.offers_dispatched)
        assert all(self.zones_price == self.pricer.zones_price)

        return pn.template.VanillaTemplate(
            main=[
                pn.Row(
                    pn.Card(
                        pn.Column(
                            pn.Row(
                                pn.widgets.Button.from_param(self.param.submit),
                                pn.widgets.Button.from_param(self.param.reset),
                            ),
                            pn.widgets.Tabulator.from_param(
                                self.param.offers_pending,
                                show_index=False,
                                formatters={
                                    Offers.price: price_usd_per_mwh.formatter,
                                    Offers.quantity: power_megawatts.formatter,
                                },
                                text_align={
                                    Offers.price: price_usd_per_mwh.align,
                                    Offers.quantity: power_megawatts.align,
                                },
                            ),
                        ),
                        title="Offers",
                    ),
                    pn.Card(
                        pn.Tabs(
                            tabulator_item(
                                self.param.offers_committed,
                                show_index=False,
                                formatters={
                                    Offers.price: price_usd_per_mwh.formatter,
                                    Offers.quantity: power_megawatts.formatter,
                                },
                                text_align={
                                    Offers.price: price_usd_per_mwh.align,
                                    Offers.quantity: power_megawatts.align,
                                },
                            ),
                            tabulator_item(
                                self.param.lines_flow,
                                formatters={
                                    LinesFlow.quantity: power_megawatts.formatter,
                                },
                                text_align={
                                    LinesFlow.quantity: power_megawatts.align,
                                },
                            ),
                            tabulator_item(
                                self.param.offers_dispatched,
                                show_index=False,
                                formatters={
                                    OffersDispatched.quantity_dispatched: power_megawatts.formatter,
                                },
                                text_align={
                                    OffersDispatched.quantity_dispatched: power_megawatts.align,
                                },
                            ),
                            tabulator_item(
                                self.param.zones_price,
                                formatters={
                                    ZonesPrice.price: price_usd_per_mwh.formatter,
                                },
                                text_align={
                                    ZonesPrice.price: price_usd_per_mwh.align,
                                },
                            ),
                        ),
                        title="Results",
                    ),
                )
            ],
            sidebar=[],
            sidebar_width=0,
            title="Mini-ISO: Auction / ISO",
        )
