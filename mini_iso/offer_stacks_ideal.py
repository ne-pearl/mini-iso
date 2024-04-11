from __future__ import annotations
import dataclasses
from typing import TypeAlias
import altair as alt
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from pandera import DataFrameModel, Field
from pandera.typing import DataFrame, Index, Series
from mini_iso.dataframes import (
    Generators,
    Offers,
    PowerMW,
    MoneyUSDPerMW,
    Zones,
)
from mini_iso.clearance import OffersOutput

Self: TypeAlias = None
LOAD_KEY = "Load"
PRICE_LOCAL_KEY = "Local Price"


class CumulativeOffers(DataFrameModel):
    """Schema for offer data."""

    generator: Index[str]
    tranche: Index[str]
    quantity_left: Series[PowerMW] = Field(coerce=True)
    quantity_right: Series[PowerMW] = Field(coerce=True)
    price_lower: Series[MoneyUSDPerMW] = Field(coerce=True)
    price_upper: Series[MoneyUSDPerMW] = Field(coerce=True)

    class Config:
        multiindex_name = "offer"
        multiindex_strict = True


@dataclasses.dataclass(frozen=True, slots=True)
class OfferStack:
    """Offer stack data"""

    cumulative_data: DataFrame[CumulativeOffers]
    load: PowerMW
    marginal_price: MoneyUSDPerMW

    @classmethod
    def from_offers(
        cls,
        offers: DataFrame[Offers],
        load: PowerMW,
    ) -> OfferStack:
        """Initialize from offers."""

        offers_by_price: pd.DataFrame = offers.sort_values(
            by=Offers.price, ascending=True
        )

        cumsum_right: NDArray[np.double] = np.cumsum(
            offers_by_price[Offers.quantity].values.tolist()
        )
        cumsum_left: NDArray[np.double] = np.insert(cumsum_right[:-1], 0, values=0.0)
        where_inside: NDArray[np.signedinteger] = np.logical_and(
            cumsum_left <= load,
            cumsum_right >= load,
        ).nonzero()[0]
        sorted_offer_prices: NDArray[np.double] = offers_by_price[Offers.price].values
        marginal_price = float(
            np.nan
            if len(where_inside) == 0
            else sorted_offer_prices[min(where_inside)].tolist()
        )

        offers_by_price[CumulativeOffers.quantity_left] = cumsum_left
        offers_by_price[CumulativeOffers.quantity_right] = cumsum_right
        offers_by_price[CumulativeOffers.price_lower] = 0.0
        offers_by_price[CumulativeOffers.price_upper] = sorted_offer_prices

        return cls(
            cumulative_data=DataFrame[CumulativeOffers](offers_by_price),
            load=load,
            marginal_price=marginal_price,
        )

    def plot(
        self,
        color_field: str,
        aggregate_load_color: str = "black",
        marginal_price_color: str = "red",
        price_axis_format: str = "$.0f",
        price_axis_title="marginal price",
        quantity_axis_title: str = "quantity [MW]",
    ) -> alt.LayerChart:
        """Produces plot of an offer stack."""

        offers_by_price: DataFrame[CumulativeOffers] = self.cumulative_data
        load: PowerMW = self.load
        marginal_price: MoneyUSDPerMW = self.marginal_price

        offers_chart = alt.Chart(offers_by_price.reset_index())
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
                x=alt.X(CumulativeOffers.quantity_left),
                x2=alt.X2(CumulativeOffers.quantity_right),
                y=alt.Y(CumulativeOffers.price_lower).scale(domainMin=0.0),
                y2=alt.Y2(CumulativeOffers.price_upper),
                color=alt.Color(color_field),
                tooltip=[
                    Offers.generator,
                    Offers.tranche,
                    OffersOutput.zone,
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
    def from_offers_by_zone(
        cls,
        offers: DataFrame[OffersOutput],
        zones: DataFrame[Zones],
    ) -> dict[str, OfferStack]:
        """Generate plots of offer stacks by bus"""
        assert "zone" in offers.columns
        offers_with_zone = offers
        return {
            zone: cls.from_offers(
                offers=DataFrame[OffersOutput](zone_offers),
                load=zones.at[zone, Zones.load],
            )
            for zone, zone_offers in offers_with_zone.groupby(OffersOutput.zone)
        }
