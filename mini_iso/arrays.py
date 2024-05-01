#!/usr/bin/env python3
from __future__ import annotations
import dataclasses
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pandera.typing import Series
import scipy
import scipy.sparse
from mini_iso.clearance import Status, clear_auction
from mini_iso.miscellaneous import DATASETS_ROOT_PATH
from mini_iso.typing_ import (
    AngleRadians,
    GeneratorId,
    Generators,
    Input,
    Lines,
    LinesSolution,
    Offers,
    OffersSolution,
    PaymentUSDPerH,
    PowerMW,
    PriceUSDPerMWh,
    Solution,
    SusceptanceS,
    TrancheId,
    ZoneId,
    Zones,
    ZonesSolution,
)


@dataclasses.dataclass(frozen=True, slots=True)
class Arrays:

    # Integer indexing for incidence matrices
    generators_index: Series[GeneratorId]
    lines_index: Series[tuple[ZoneId, ZoneId]]
    offers_index: Series[tuple[GeneratorId, TrancheId]]
    zones_index: Series[ZoneId]

    # Incidence matrices
    zones_lines_incidence: scipy.sparse.sparray
    zones_offers_incidence: scipy.sparse.sparray

    # Problem data
    base_power: float
    generators_capacity: Series[PowerMW]
    lines_capacity: Series[PowerMW]
    lines_susceptance: Series[SusceptanceS]
    offers_price: Series[PriceUSDPerMWh]
    offers_quantity: Series[PowerMW]
    zones_load: Series[PowerMW]

    # Solution arrays
    objective: PaymentUSDPerH
    offers_dispatched: Series[PowerMW]
    lines_quantity: Series[PowerMW]
    zones_angle: Series[AngleRadians]
    zones_price: Series[PriceUSDPerMWh]

    # Dual variables on variable bounds
    reference_angle_coef: float
    lines_angles_dual: Series[float]
    lines_quantity_lb_coef: Series[PriceUSDPerMWh]
    lines_quantity_ub_coef: Series[PriceUSDPerMWh]
    offers_dispatched_dual_lb: Series[PriceUSDPerMWh]
    offers_dispatched_dual_ub: Series[PriceUSDPerMWh]


    @classmethod
    def init(cls, inputs: Input, solution: Solution) -> Arrays:

        # Use contiguous integer indexing for compatibility with scipy arrays
        generators = inputs.generators.reset_index()
        lines = inputs.lines.reset_index()
        offers = inputs.offers.reset_index()
        zones = inputs.zones.reset_index()

        num_lines: int = lines.index.size
        num_offers: int = offers.index.size
        num_zones: int = zones.index.size

        # generators_zone: Series[ZoneId] = generators[Generators.zone]
        # offers_generator: Series[GeneratorId] = offers[Offers.generator]
        # offers_tranche: Series[TrancheId] = offers[Offers.tranche]
        # offers_cost: Series[PriceUSDPerMWh] = offers[Offers.price]
        # offers_quantity: Series[PowerMW] = offers[Offers.quantity]
        lines_from: Series[ZoneId] = lines[Lines.zone_from]
        lines_to: Series[ZoneId] = lines[Lines.zone_to]
        # lines_capacity: Series[PowerMW] = lines[Lines.capacity]
        # lines_susceptance: Series[SusceptanceS] = lines[Lines.susceptance]
        # zones_name: Series[ZoneId] = zones[Zones.name]
        # zones_load: Series[PowerMW] = zones[Zones.load]
        offers_zone: Series[ZoneId] = pd.merge(
            left=offers[Offers.generator],
            right=generators[[Generators.name, Generators.zone]],
            how="left",
            left_on=Offers.generator,
            right_on=Generators.name,
        )[Generators.zone]

        # generators_index = Series[int](
        #     data=generators.index.values,
        #     index=generators[Generators.name],
        # )
        # lines_index = Series[int](
        #     data=lines.index.values,
        #     index=lines[[Lines.zone_from, Lines.zone_to]],
        # )
        # offers_index = Series[int](
        #     data=offers.index.values,
        #     index=offers[[Offers.generator, Offers.tranche]],
        # )
        zones_index = Series[int](
            data=zones.index.values,
            index=zones[Zones.name],
        )

        zones_lines_incidence = scipy.sparse.csc_array(
            (
                np.concatenate(
                    (
                        np.tile(-1.0, num_lines),  # out
                        np.tile(+1.0, num_lines),  # on
                    ),
                ),
                (
                    np.concatenate(
                        (
                            zones_index.loc[lines_from].values,  # out
                            zones_index.loc[lines_to].values,  # on
                        ),
                    ),
                    np.concatenate(
                        (
                            lines.index.values,  # out
                            lines.index.values,  # on
                        )
                    ),
                ),
            ),
            shape=(num_zones, num_lines),
        )

        zones_offers_incidence = scipy.sparse.csc_array(
            (
                np.ones(offers.index.size),
                (
                    zones_index.loc[offers_zone].values,
                    offers.index.values,
                ),
            ),
            shape=(num_zones, num_offers),
        )

        return cls(
            # Integer indexing for incidence matrices
            generators_index=Series[GeneratorId](generators[Generators.name]),
            lines_index=Series[tuple[ZoneId, ZoneId]](
                zip(
                    lines[Lines.zone_from],
                    lines[Lines.zone_to],
                )
            ),
            offers_index=Series[tuple[GeneratorId, TrancheId]](
                zip(
                    offers[Offers.generator],
                    offers[Offers.tranche],
                )
            ),
            zones_index=Series[ZoneId](zones[Zones.name]),
            # Incidence matrices
            zones_lines_incidence=zones_lines_incidence,
            zones_offers_incidence=zones_offers_incidence,
            # Problem data
            base_power=inputs.base_power,
            generators_capacity=Series[PowerMW](generators[Generators.capacity]),
            lines_capacity=Series[PowerMW](lines[Lines.capacity]),
            lines_susceptance=Series[float](lines[Lines.susceptance]),
            offers_price=Series[PriceUSDPerMWh](offers[Offers.price]),
            offers_quantity=Series[PriceUSDPerMWh](offers[Offers.quantity]),
            zones_load=Series[PowerMW](zones[Zones.load]),
            # Solution arrays
            objective=solution.objective,
            offers_dispatched=Series[PowerMW](
                solution.offers[OffersSolution.quantity_dispatched]
            ),
            lines_quantity=Series[PowerMW](solution.lines[LinesSolution.quantity]),
            zones_angle=Series[AngleRadians](solution.zones[ZonesSolution.angle]),
            # Dual variables
            zones_price=Series[PriceUSDPerMWh](solution.zones[ZonesSolution.price]),
            reference_angle_coef=solution.reference_angle_coef,
            lines_angles_dual=solution.lines[LinesSolution.angle_dual],
            lines_quantity_lb_coef=solution.lines[LinesSolution.quantity_lb_coef],
            lines_quantity_ub_coef=solution.lines[LinesSolution.quantity_ub_coef],
            offers_dispatched_dual_lb=solution.offers[OffersSolution.quantity_lb_coef],
            offers_dispatched_dual_ub=solution.offers[OffersSolution.quantity_ub_coef],
        )
