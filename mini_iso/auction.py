from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from pandera.typing import Series
import panel as pn
import param as pm
from mini_iso.dataframes import Offers
from mini_iso.panel_helpers import tabulator_item
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
        # assert self.param.lines_flow is pricer.param.lines_flow
        # assert self.param.offers_committed is pricer.param.offers
        # assert self.param.offers_dispatched is pricer.param.offers_dispatched
        # assert self.param.zones_price is pricer.param.zones_price

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
                            ),
                        ),
                        title="Offers",
                    ),
                    pn.Card(
                        pn.Tabs(
                            tabulator_item(
                                self.param.offers_committed,
                                show_index=False,
                            ),
                            tabulator_item(self.param.lines_flow),
                            tabulator_item(
                                self.param.offers_dispatched,
                                show_index=False,
                            ),
                            tabulator_item(self.param.zones_price),
                        ),
                        title="Results",
                    ),
                )
            ],
            sidebar=[],
            sidebar_width=0,
            title="Mini-ISO: Auction / ISO",
        )


class Bidder(pn.viewable.Viewer):

    auction = pm.ClassSelector(class_=Auction, instantiate=False, label="Auction")
    generator_name = pm.Selector(label="Generator")
    offers_drafted = pm.DataFrame(label="Draft Offers")
    offers_pending = pm.DataFrame(label="Pending Offers")
    offers_committed = pm.DataFrame(label="Committed Offers")
    offers_dispatched = pm.DataFrame(label="Offers Dispatched")
    submit = pm.Event(label="Submit")
    reset = pm.Event(label="Reset")

    zones_price = pm.DataFrame(label="Zone Prices", allow_refs=True, instantiate=False)

    def __init__(self, auction: Auction, **params):
        super().__init__(
            **params,
            auction=auction,
            zones_price=auction.param.zones_price,  # direct link
        )
        names: list[str] = auction.offers_pending[Offers.generator].unique().tolist()
        assert len(names) != 0
        self.param.generator_name.objects = names
        self.generator_name = names[0]

    def _rows(self) -> Series[np.bool]:
        generator_names: Series[str] = self.auction.offers_pending[Offers.generator]
        mask: NDArray[np.bool] = generator_names == self.generator_name
        return Series(mask, index=generator_names.index)

    @pn.depends("generator_name", "reset", watch=True)
    def _on_reset(self) -> None:
        # TODO: Should we roll back to previously submitted or -committed?
        rows: Series[np.bool] = self._rows()
        self.offers_drafted = self.auction.offers_pending[rows]
        self.offers_pending = self.auction.offers_pending[rows]
        self.offers_committed = self.auction.offers_committed[rows]
        # print(">" * 80)
        # print(rows)
        # print("dispatched:\n", self.auction.offers_dispatched)
        # print("<" * 80)
        self.offers_dispatched = self.auction.offers_dispatched[rows]

    @pn.depends("submit", watch=True)
    def _on_submit(self) -> None:
        buffer = self.auction.offers_pending.copy()
        buffer[self._rows()] = self.offers_drafted
        self.auction.offers_pending = buffer

    @pn.depends("auction.offers_pending", watch=True)
    def _on_auction_offers_pending(self) -> None:
        self.offers_pending = self.auction.offers_pending[self._rows()]

    @pn.depends("auction.offers_committed", watch=True)
    def _on_auction_offers_committed(self) -> None:
        self.offers_committed = self.auction.offers_committed[self._rows()]

    @pn.depends("auction.offers_dispatched", watch=True)
    def _on_auction_offers_dispatched(self) -> None:
        self.offers_dispatched = self.auction.offers_dispatched[self._rows()]

    def __panel__(self) -> pn.viewable.Viewable:
        # FIXME: Breaks encapsulation
        return pn.template.VanillaTemplate(
            main=[
                pn.Column(
                    pn.widgets.Select.from_param(
                        self.param.generator_name,
                        name="Generator",
                    ),
                    pn.Row(
                        pn.Column(
                            pn.Card(
                                pn.Column(
                                    pn.widgets.Tabulator.from_param(
                                        self.param.offers_drafted
                                    ),
                                    pn.Row(
                                        pn.widgets.Button.from_param(self.param.submit),
                                        pn.widgets.Button.from_param(self.param.reset),
                                    ),
                                ),
                                title="Draft Offers",
                            ),
                            pn.Card(
                                pn.widgets.Tabulator.from_param(
                                    self.param.offers_pending,
                                    disabled=True,
                                    show_index=False,
                                ),
                                title="Submitted Offers",
                            ),
                        ),
                        pn.Card(
                            pn.Column(
                                pn.Card(
                                    pn.widgets.Tabulator.from_param(
                                        self.param.offers_committed,
                                        show_index=False,
                                    ),
                                    title="Committed Offers",
                                ),
                                pn.Card(
                                    pn.widgets.Tabulator.from_param(
                                        self.param.offers_dispatched,
                                        show_index=False,
                                    ),
                                    title="Dispatched Offers",
                                ),
                                pn.Card(
                                    pn.widgets.Tabulator.from_param(
                                        self.param.zones_price
                                    ),
                                    title="Zone Prices",
                                ),
                            ),
                            title="Results",
                        ),
                    ),
                )
            ],
            sidebar=[],
            sidebar_width=0,
            title="Mini-ISO: Bidding / Generator",
        )
