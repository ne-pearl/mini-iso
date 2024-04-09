import enum
from dataclasses import dataclass, fields
import math
import sys
from typing import Any, TypeAlias

import gurobipy as grb
import numpy as np
import pandas as pd
from pandera import DataFrameModel, Field
from pandera.api.pandas import model_config
from pandera.typing import DataFrame, Index, Series
from mini_iso.dataframes import (
    Input,
    Generators,
    Lines,
    MoneyUSDPerMW,
    Offers,
    SpatialCoordinate,
    Zones,
)


class Status(enum.Enum):
    INFEASIBLE = "infeasible problem"
    OPTIMAL = "optimal solution"
    UNBOUNDED = "unbounded solution"
    UNKNOWN = "uknown or unsupported"


# Tolerance for detection of binding constraints
BIND_TOL = 1.0 / 100.0

GeneratorId: TypeAlias = str
LineId: TypeAlias = int
TranchId: TypeAlias = str
ZoneId: TypeAlias = str
OfferId: TypeAlias = tuple[GeneratorId, TranchId]
Fraction: TypeAlias = float
PowerMW: TypeAlias = float
MoneyUSD: TypeAlias = float
Susceptance: TypeAlias = float  # (1/reactance)


class GeneratorsSolution(DataFrameModel):
    name: Index[str]
    # cost: Series[MoneyUSD]
    # revenue: Series[MoneyUSD]
    # benefit: Series[MoneyUSD]
    # output: Series[PowerMW]


class LinesSolution(DataFrameModel):
    # FIXME: For consistency, Index[int] should be Index[str]
    name: Index[int] = Field(unique=True)
    quantity: Series[PowerMW] = Field(coerce=True)
    # slack: Series[PowerMW] = Field(coerce=True)


class OffersSolution(DataFrameModel):
    generator: Index[str]
    tranche: Index[str]
    quantity_dispatched: Series[PowerMW] = Field(coerce=True)
    # is_marginal: Series[bool] = Field(coerce=True)


class ZonesSolution(DataFrameModel):
    name: Index[str] = Field(unique=True)
    price: Series[MoneyUSD] = Field(coerce=True)


@dataclass(frozen=True, slots=True)
class Solution:
    # generators: DataFrame[GeneratorsSolution]
    lines: DataFrame[LinesSolution]
    offers: DataFrame[OffersSolution]
    zones: DataFrame[ZonesSolution]


def solve(
    input_: Input,
    base_power: PowerMW = 1000,  # big_m: float = 10000,
) -> tuple[Status, Solution | None]:

    # Allow index levels and columns to be referenced uniformly
    generators_df = input_.generators.reset_index()
    offers_df = input_.offers.reset_index()
    lines_df = input_.lines.reset_index()
    zones_df = input_.zones.reset_index()

    # Variable sets
    # FIXME: Ultimately, replace each List with Set
    G: list[GeneratorId] = list(generators_df[Generators.name])
    L: list[LineId] = list(lines_df[Lines.name])
    Z: list[ZoneId] = list(zones_df[Zones.name])

    Gz1: dict[GeneratorId, ZoneId] = dict(
        zip(generators_df[Generators.name], generators_df[Generators.zone])
    )
    Tg: dict[GeneratorId, list[TranchId]] = {g: [] for g in G}
    Tg.update(
        (str(g), list(group[Offers.tranche]))
        for g, group in offers_df.groupby(Offers.generator)
    )
    # FIXME: Remove field "is_included"; inherited from old code
    is_included = (
        generators_df[Generators.is_included]
        if Generators.is_included in generators_df.columns
        else slice(None)
    )
    Gd: list[GeneratorId] = list(
        generators_df[Generators.name][is_included]
    )  # chosen generators
    GP: list[GeneratorId] = sorted(set(G).difference(Gd))  # complement of chosen

    # Zone mappings
    Gz: dict[ZoneId, list[GeneratorId]] = {z: [] for z in Z}
    Gz.update(
        (str(z), list(group[Generators.name]))
        for z, group in generators_df.groupby(by=Generators.zone)
    )

    Lo: dict[ZoneId, list[LineId]] = {z: [] for z in Z}
    Lo.update(
        (str(z), list(group[Lines.name]))
        for z, group in lines_df.groupby(Lines.zone_from)
    )

    Le: dict[ZoneId, list[LineId]] = {z: [] for z in Z}
    Le.update(
        (str(z), list(group[Lines.name]))
        for z, group in lines_df.groupby(Lines.zone_to)
    )

    Czl: dict[LineId, tuple[ZoneId, ZoneId]] = dict(
        zip(
            lines_df[Lines.name],
            zip(lines_df[Lines.zone_from], lines_df[Lines.zone_to]),
        )
    )  # line -> adjacent zones

    B: dict[LineId, Susceptance] = dict(
        zip(lines_df[Lines.name], lines_df[Lines.susceptance])
    )

    # Parameters:
    tranche_pmax: dict[OfferId, PowerMW] = {
        (str(g), str(t)): p_max
        for g, group in offers_df.groupby(Offers.generator)
        for t, p_max in zip(group[Offers.tranche], group[Offers.quantity])
    }  # max output for generator offers

    # FIXME: Use these updated bounds in place of extra constraints
    # pmax.update({(g, t): 0.0 for g in GP for t in Tg[g]})  # excluded units

    tranche_cost: dict[OfferId, MoneyUSD] = {
        (str(g), str(t)): price
        for g, group in offers_df.groupby(Offers.generator)
        for t, price in zip(group[Offers.tranche], group[Offers.price])
    }

    line_capacity: dict[LineId, PowerMW] = dict(
        zip(lines_df[Lines.name], lines_df[Lines.capacity])
    )
    zone_load: dict[ZoneId, PowerMW] = dict(
        zip(zones_df[Zones.name], zones_df[Zones.load])
    )

    # Formulate and solve the pricing model
    model = grb.Model()

    # Declaring Variables
    p: grb.tupledict = model.addVars(
        tranche_pmax.keys(),
        lb=0.0,
        ub=tranche_pmax,
        name="power_output",
        vtype=grb.GRB.CONTINUOUS,
    )
    w: grb.tupledict = model.addVars(
        L,
        lb={l_: -capacity for l_, capacity in line_capacity.items()},
        ub=line_capacity,
        name="power_on_line",
        vtype=grb.GRB.CONTINUOUS,
    )
    # alpha: grb.tupledict = model.addVars(
    #     Z, lb=0.0, name="generation_slack", vtype=grb.GRB.CONTINUOUS
    # )
    theta: grb.tupledict = model.addVars(
        Z, lb=-math.pi, ub=+math.pi, name="voltage_angle", vtype=grb.GRB.CONTINUOUS
    )

    sum_ = grb.quicksum

    # Objective function
    model.setObjective(
        sum_(tranche_cost[g, t] * p[g, t] for g, t in tranche_cost.keys())
        # + sum_(big_m * alpha[z] for z in Z)
        ,
        grb.GRB.MINIMIZE,
    )

    # Constraints
    power_flow_constraints: grb.tupledict = model.addConstrs(
        (
            sum_(p[g, t] for g in Gz[z] for t in Tg[g])
            + sum_(w[le] for le in Le.get(z, []))
            - sum_(w[lo] for lo in Lo.get(z, []))
            # + alpha[z]
            == zone_load[z]
            for z in Z
        ),
        name="power_flow",
    )
    model.addConstrs(
        (
            base_power * B[l_] * (theta[zone_pair[0]] - theta[zone_pair[1]]) == w[l_]
            for l_ in L
            if (zone_pair := Czl[l_]) or exit("unreachable")
        ),
        name="voltage_angle",
    )
    model.addConstr(
        theta[Czl[1][0]] == 0.0,
        name="angle_ref",
    )
    # FIXME: Replace with zero bounds on excluded generators
    model.addConstrs(
        (p[g, t] == 0.0 for g in GP for t in Tg[g]),
        name="excluded_generators",
    )

    model.setParam(grb.GRB.Param.OutputFlag, 0)
    model.optimize()

    if model.Status == grb.GRB.INF_OR_UNBD:
        # Turn presolve off to determine whether model is infeasible or unbounded
        print("Infeasible or unbounded model.", file=sys.stderr)
        model.setParam(grb.GRB.Param.Presolve, 0)
        model.optimize()

    status: Status = {
        grb.GRB.INFEASIBLE: Status.INFEASIBLE,
        grb.GRB.OPTIMAL: Status.OPTIMAL,
        grb.GRB.UNBOUNDED: Status.UNBOUNDED,
    }.get(model.Status, Status.UNKNOWN)

    if status is Status.OPTIMAL:
        print(f"... optimal objective: {model.ObjVal:g}")
    else:
        print(f"... stopped with status {model.Status}: {status.value}")
        return status, None

    assert status is Status.OPTIMAL

    price = pd.Series(
        {z: constr.getAttr("Pi") for z, constr in power_flow_constraints.items()}
    )
    price.index.name = Zones.name

    # # FIXME: Move to post-processing
    # # Benefit to generators
    # cost: dict[GeneratorId, MoneyUSD] = {
    #     g: sum(p[g, t].x for t in Tg[g]) * c
    #     for g, c in zip(generators_df[Generators.name], generators_df[Generators.cost])
    # }
    # punishment: MoneyUSD = 0.0  # FIXME: punishment price?
    # revenue: dict[GeneratorId, MoneyUSD] = {
    #     g: price[Gz1[g]] * sum(p[g, t].x for t in Tg[g]) for g in G
    # }
    # benefit_all: dict[GeneratorId, MoneyUSD] = {
    #     g: revenue[g] - cost[g] - punishment for g in G
    # }
    # benefit_selected = {g: benefit_all[g] for g in Gd}

    line_power: Series[PowerMW] = pd.Series({l_: w[l_].x for l_ in L})
    line_power.index.name = Lines.name

    # Dispatched output for each generator over its tranches
    offer_index_levels: list[str] = [Offers.generator, Offers.tranche]
    dispatched = pd.DataFrame.from_records(
        [
            {
                Offers.generator: g,
                Offers.tranche: t,
                Offers.quantity: pgt.x,
            }
            for (g, t), pgt in p.items()
        ],
    ).set_index(offer_index_levels)

    # Sanity check
    total_load: PowerMW = input_.zones[ZonesOutput.load].sum()
    total_generation: PowerMW = dispatched[Offers.quantity].sum()
    mismatch: PowerMW = total_generation - total_load
    assert abs(mismatch) / total_load < 1e-6

    # # FIXME: Gymnastics to get a named index!
    # generators_temporary = pd.DataFrame(
    #     {
    #         GeneratorsSolution.cost: cost,
    #         GeneratorsSolution.revenue: revenue,
    #         GeneratorsSolution.benefit: benefit_selected,
    #     }
    # )
    # generators_temporary.index.name = Generators.name

    return status, Solution(
        # generators=DataFrame[GeneratorsSolution](generators_temporary),
        lines=DataFrame[LinesSolution](
            {
                LinesSolution.quantity: line_power,
                # LinesSolution.slack: line_slack,
            }
        ),
        zones=DataFrame[ZonesSolution](
            {
                ZonesSolution.price: price,
            }
        ),
        offers=DataFrame[OffersSolution](
            {
                OffersSolution.quantity_dispatched: dispatched[Offers.quantity],
            }
        ),
    )


class GeneratorsOutput(GeneratorsSolution):
    capacity: Series[PowerMW] = Field(coerce=True)
    zone: Series[ZoneId]
    dispatched: Series[PowerMW] = Field(coerce=True)
    utilization: Series[Fraction] = Field(coerce=True)


class LinesOutput(LinesSolution):
    zone_from: Series[ZoneId]
    zone_to: Series[ZoneId]
    susceptance: Series[Susceptance] = Field(coerce=True)
    abs_flow: Series[PowerMW] = Field(coerce=True)
    capacity: Series[PowerMW] = Field(coerce=True)
    slack: Series[PowerMW] = Field(coerce=True)
    utilization: Series[Fraction]
    is_critical: Series[bool]
    x_from: Series[SpatialCoordinate] = Field(coerce=True)
    y_from: Series[SpatialCoordinate] = Field(coerce=True)
    x_to: Series[SpatialCoordinate] = Field(coerce=True)
    y_to: Series[SpatialCoordinate] = Field(coerce=True)
    x_mid: Series[SpatialCoordinate] = Field(coerce=True)
    y_mid: Series[SpatialCoordinate] = Field(coerce=True)


class OffersOutput(DataFrameModel):

    class Config(model_config.BaseConfig):
        multiindex_name = "offer"
        multiindex_strict = True
        unique_column_names: bool = True

    generator: Index[str]
    tranche: Index[str]
    zone: Series[ZoneId]
    price: Series[MoneyUSD] = Field(coerce=True)
    quantity: Series[PowerMW] = Field(coerce=True)
    quantity_dispatched: Series[PowerMW] = Field(coerce=True)
    utilization: Series[Fraction] = Field(coerce=True)
    is_marginal: Series[bool] = Field(coerce=True)


class ZonesOutput(ZonesSolution):
    load: Series[PowerMW] = Field(coerce=True)
    capacity: Series[PowerMW] = Field(coerce=True)
    dispatched: Series[PowerMW] = Field(coerce=True)
    utilization: Series[PowerMW] = Field(coerce=True)
    x: Series[SpatialCoordinate] = Field(coerce=True)
    y: Series[SpatialCoordinate] = Field(coerce=True)


@dataclass(frozen=True, slots=True)
class Output:

    generators: DataFrame[GeneratorsOutput]
    lines: DataFrame[LinesOutput]
    offers: DataFrame[OffersOutput]
    zones: DataFrame[ZonesOutput]


def post_process(input_: Input, solution: Solution) -> Output:

    all_offers_data = pd.concat((input_.offers, solution.offers), axis="columns")
    offers_by_generator = all_offers_data.groupby(Offers.generator)
    generators_dispatched: Series[PowerMW] = offers_by_generator[
        OffersSolution.quantity_dispatched
    ].sum()
    assert generators_dispatched.index.name == Offers.generator
    generators_dispatched.index.name = GeneratorsOutput.name
    generators_capacity: Series[PowerMW] = input_.generators[Generators.capacity]

    generators = DataFrame[GeneratorsOutput](
        {
            **dict(solution.generators),
            GeneratorsOutput.capacity: generators_capacity,
            GeneratorsOutput.cost: input_.generators[Generators.cost],
            GeneratorsOutput.zone: input_.generators[Generators.zone],
            GeneratorsOutput.dispatched: generators_dispatched,
            GeneratorsOutput.utilization: generators_dispatched / generators_capacity,
        }
    )

    x_of_zones = input_.zones[Zones.x]
    y_of_zones = input_.zones[Zones.y]
    zone_from_of_lines = input_.lines[Lines.zone_from]
    zone_to_of_lines = input_.lines[Lines.zone_to]

    lines_capacity: Series[PowerMw] = input_.lines[Lines.capacity]
    lines_slack: Series[PowerMW] = (
        lines_capacity - solution.lines[LinesSolution.quantity].abs()
    )
    lines_utilization: Series[Fraction] = 1.0 - lines_slack / lines_capacity
    lines_is_critical: Series[bool] = (BIND_TOL - 1.0) <= lines_utilization

    lines = DataFrame[LinesOutput](
        {
            **dict(solution.lines),
            LinesOutput.abs_flow: solution.lines[LinesSolution.quantity].abs(),
            LinesOutput.capacity: lines_capacity,
            LinesOutput.slack: lines_slack,
            LinesOutput.utilization: lines_utilization,
            LinesOutput.is_critical: lines_is_critical,
            LinesOutput.x_from: x_of_zones[zone_from_of_lines].values,
            LinesOutput.y_from: y_of_zones[zone_from_of_lines].values,
            LinesOutput.x_to: x_of_zones[zone_to_of_lines].values,
            LinesOutput.y_to: y_of_zones[zone_to_of_lines].values,
        }
    )

    generator_of_offers: Index[GeneratorId] = input_.offers.index.get_level_values(
        Offers.generator
    )
    zone_of_generators: Series[ZoneId] = input_.generators[Generators.zone]

    # Utilization of each offer
    offers_utilization: Series[Fraction] = (
        solution.offers[OffersSolution.quantity_dispatched]
        / input_.offers[Offers.quantity]
    )
    # fmt: off
    offers_is_marginal: Series[bool] = (
        (BIND_TOL <= offers_utilization) 
                  & (offers_utilization <= 1.0 - BIND_TOL)
    )
    # fmt: on

    offers = DataFrame[OffersOutput](
        {
            **dict(solution.offers),
            OffersOutput.zone: zone_of_generators[generator_of_offers].values,
            OffersOutput.utilization: offers_utilization,
            OffersOutput.is_marginal: offers_is_marginal,
        }
    )

    # Sum dispatched output over each zone's generators:
    zones_generation_temporary: Series[PowerMW] = generators.groupby(Generators.zone)[
        [
            GeneratorsOutput.capacity,
            GeneratorsOutput.dispatched,
        ]
    ].sum()

    # Zones needn't all have generators: Update index and replace NaN with 0.0
    zones_generation: pd.DataFrame = pd.merge(
        left=zones_generation_temporary,
        left_index=True,
        right=input_.zones[[]],  # index only
        right_index=True,
        how="right",
    ).fillna(0.0)

    zones_dispatched: Series[PowerMW] = zones_generation[GeneratorsOutput.dispatched]
    zones_capacity: Series[PowerMW] = zones_generation[GeneratorsOutput.capacity]
    # Replace 0/0=NaN with 0
    zones_utilization: Series[Fraction] = (zones_dispatched / zones_capacity).fillna(
        0.0
    )

    zones = DataFrame[ZonesOutput](
        {
            **dict(solution.zones),
            ZonesOutput.load: input_.zones[Zones.load],
            ZonesOutput.dispatched: zones_dispatched,
            ZonesOutput.capacity: zones_capacity,
            ZonesOutput.utilization: zones_utilization,
            ZonesOutput.x: input_.zones[Zones.x],
            ZonesOutput.y: input_.zones[Zones.y],
        }
    )

    return Output(
        generators=generators,
        lines=lines,
        offers=offers,
        zones=zones,
    )
