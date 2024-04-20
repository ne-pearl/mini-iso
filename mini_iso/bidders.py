from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pandera.typing import Series
import panel as pn
import param as pm
from mini_iso.typing import (
    OFFERS_INDEX_LABELS,
    Generators,
    Offers,
    OffersSummary,
    PaymentUSDPerH,
    PowerMW,
    ZonesPrice,
)
from mini_iso.miscellaneous import (
    fraction_percentage,
    payment_usd_per_h,
    price_usd_per_mwh,
    power_megawatts,
)
from mini_iso.auction import Auction


class Bidder(pn.viewable.Viewer):
    auction = pm.ClassSelector(class_=Auction, instantiate=False, label="Auction")
    generator_name = pm.Selector(label="Generator")
    offers_drafted = pm.DataFrame(label="Draft Offers")
    offers_pending = pm.DataFrame(label="Pending Offers")
    offers_committed = pm.DataFrame(label="Committed Offers")
    offers_dispatched = pm.DataFrame(label="Offers Dispatched")
    submit = pm.Event(label="Submit")
    reset = pm.Event(label="Reset")
    cost = pm.Number(label="Marginal Cost")
    zone = pm.String(label="Zone")
    zones_price = pm.DataFrame(label="Zone Prices", allow_refs=True, instantiate=False)
    summary = pm.DataFrame(label="Summary")

    def __init__(self, auction: Auction, **params):
        super().__init__(
            **params,
            auction=auction,
            zones_price=auction.param.zones_price,  # direct link
        )
        names: list[str] = sorted(auction.offers_pending[Offers.generator].unique())
        assert len(names) != 0
        self.param.generator_name.objects = names
        self.generator_name = names[0]
        self._update_summary()

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
        self.offers_dispatched = self.auction.offers_dispatched[rows]
        self.zone = self.auction.pricer.generators.at[
            self.generator_name,
            Generators.zone,
        ]
        self.cost = self.auction.pricer.generators.at[
            self.generator_name,
            Generators.cost,
        ]
        self._update_summary()

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
        self._clear_summary()

    @pn.depends("auction.offers_dispatched", watch=True)
    def _on_auction_offers_dispatched(self) -> None:
        self.offers_dispatched = self.auction.offers_dispatched[self._rows()]

    @pn.depends("auction.param", watch=True)
    def _on_auction_param(self) -> None:
        self._update_summary()

    def _clear_summary(self) -> None:
        """Clears computed fields of summary table."""
        self.summary[
            [
                OffersSummary.price_offered,
                OffersSummary.excess,
                OffersSummary.quantity_dispatched,
                OffersSummary.utilization,
            ]
        ] = None

    def _update_summary(self) -> None:
        """Updates summary table."""
        summary: pd.DataFrame = pd.concat(
            [
                self.offers_committed.set_index(OFFERS_INDEX_LABELS),
                self.offers_dispatched.set_index(OFFERS_INDEX_LABELS),
            ],
            axis="columns",
        ).rename(
            columns={
                Offers.quantity: OffersSummary.quantity_offered,
                Offers.price: OffersSummary.price_offered,
            }
        )
        summary[OffersSummary.price_lmp] = self.zones_price.at[
            self.zone, ZonesPrice.price
        ]
        summary[OffersSummary.excess] = (
            summary[OffersSummary.price_lmp] - summary[OffersSummary.price_offered]
        )
        summary[OffersSummary.utilization] = (
            summary[OffersSummary.quantity_dispatched]
            / summary[OffersSummary.quantity_offered]
        ).fillna(0.0)
        summary[OffersSummary.revenue] = (
            summary[OffersSummary.price_lmp]
            * summary[OffersSummary.quantity_dispatched]
        )

        OffersSummary.validate(summary)

        # Add totals
        summary.reset_index(inplace=True)
        offered_total: PowerMW = summary[OffersSummary.quantity_offered].sum()
        dispatched_total: PowerMW = summary[OffersSummary.quantity_dispatched].sum()
        revenue_total: PaymentUSDPerH = summary[OffersSummary.revenue].sum()
        # Insert totals in bottom row
        next: int = summary.index.values.max() + 1
        summary.loc[next, :] = ""
        summary.loc[next, OffersSummary.tranche] = "TOTAL"
        summary.loc[next, OffersSummary.quantity_offered] = offered_total
        summary.loc[next, OffersSummary.quantity_dispatched] = dispatched_total
        summary.loc[next, OffersSummary.utilization] = (
            dispatched_total / offered_total if offered_total != 0.0 else 0.0
        )
        summary.loc[next, OffersSummary.revenue] = revenue_total

        self.summary = summary

    def __panel__(self) -> pn.viewable.Viewable:
        # FIXME: Breaks encapsulation

        return pn.template.VanillaTemplate(
            main=[
                pn.Column(
                    pn.widgets.Select.from_param(
                        self.param.generator_name,
                        name="Generator",
                    ),
                    pn.widgets.StaticText.from_param(self.param.zone),
                    pn.widgets.StaticText.from_param(self.param.cost),
                    pn.Row(
                        pn.Column(
                            pn.Card(
                                pn.Column(
                                    pn.widgets.Tabulator.from_param(
                                        self.param.offers_drafted,
                                        formatters={
                                            Offers.price: price_usd_per_mwh.formatter,
                                            Offers.quantity: power_megawatts.formatter,
                                        },
                                        show_index=False,
                                        text_align={
                                            Offers.price: price_usd_per_mwh.align,
                                            Offers.quantity: power_megawatts.align,
                                        },
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
                                    formatters={
                                        Offers.price: price_usd_per_mwh.formatter,
                                        Offers.quantity: power_megawatts.formatter,
                                    },
                                    show_index=False,
                                    text_align={
                                        Offers.price: price_usd_per_mwh.align,
                                        Offers.quantity: power_megawatts.align,
                                    },
                                ),
                                title="Submitted Offers",
                            ),
                        ),
                        pn.Card(
                            pn.Column(
                                pn.widgets.Tabulator.from_param(
                                    self.param.summary,
                                    formatters={
                                        OffersSummary.excess: price_usd_per_mwh.formatter,
                                        OffersSummary.price_lmp: price_usd_per_mwh.formatter,
                                        OffersSummary.price_offered: price_usd_per_mwh.formatter,
                                        OffersSummary.quantity_dispatched: power_megawatts.formatter,
                                        OffersSummary.quantity_offered: power_megawatts.formatter,
                                        OffersSummary.revenue: payment_usd_per_h.formatter,
                                        OffersSummary.utilization: fraction_percentage.formatter,
                                    },
                                    disabled=True,
                                    show_index=False,
                                    text_align={
                                        OffersSummary.excess: price_usd_per_mwh.align,
                                        OffersSummary.price_lmp: price_usd_per_mwh.align,
                                        OffersSummary.price_offered: price_usd_per_mwh.align,
                                        OffersSummary.quantity_dispatched: power_megawatts.align,
                                        OffersSummary.quantity_offered: power_megawatts.align,
                                        OffersSummary.revenue: payment_usd_per_h.align,
                                        OffersSummary.utilization: fraction_percentage.align,
                                    },
                                    titles={
                                        OffersSummary.generator: "name",
                                        OffersSummary.price_lmp: "LMP",
                                        OffersSummary.price_offered: "offer",
                                        OffersSummary.quantity_dispatched: "dispatched",
                                        OffersSummary.quantity_offered: "offer",
                                    },
                                ),
                                pn.Card(
                                    pn.widgets.Tabulator.from_param(
                                        self.param.zones_price,
                                        formatters={
                                            ZonesPrice.price: price_usd_per_mwh.formatter,
                                        },
                                        disabled=True,
                                        show_index=True,
                                        text_align={
                                            ZonesPrice.price: price_usd_per_mwh.align,
                                        },
                                    ),
                                    title="Zone Prices",
                                ),
                            ),
                            title="Last Auction",
                        ),
                    ),
                )
            ],
            sidebar=[],
            sidebar_width=0,
            title="Mini-ISO: Bidding / Generator",
        )
