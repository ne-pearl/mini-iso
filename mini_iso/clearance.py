import enum
from dataclasses import dataclass
import math
import sys
import gurobipy as grb
import pandas as pd
from pandera.typing import DataFrame, Series
from mini_iso.typing import (
    OFFERS_INDEX_LABELS,
    Input,
    Generators,
    GeneratorId,
    LineId,
    Lines,
    LinesSolution,
    MoneyUSDPerMW,
    OfferId,
    Offers,
    OffersSolution,
    PowerMW,
    Susceptance,
    TrancheId,
    ZoneId,
    Zones,
    ZonesOutput,
    ZonesSolution,
)


class Status(enum.Enum):
    INFEASIBLE = "infeasible problem"
    OPTIMAL = "optimal solution"
    UNBOUNDED = "unbounded solution"
    UNKNOWN = "uknown or unsupported"


@dataclass(frozen=True, slots=True)
class Solution:
    # generators: DataFrame[GeneratorsSolution]
    lines: DataFrame[LinesSolution]
    offers: DataFrame[OffersSolution]
    zones: DataFrame[ZonesSolution]


def clear_auction(
    inputs: Input,
    base_power: PowerMW = 1000,
) -> tuple[Status, Solution | None]:
    """Solves linear pricing problem for real-time market."""

    # Allow index levels and columns to be referenced uniformly
    generators_df = inputs.generators.reset_index()
    offers_df = inputs.offers.reset_index()
    lines_df = inputs.lines.reset_index()
    zones_df = inputs.zones.reset_index()

    # Variable sets
    # FIXME: Ultimately, replace each List with Set
    G: list[GeneratorId] = list(generators_df[Generators.name])
    L: list[LineId] = list(lines_df[Lines.name])
    Z: list[ZoneId] = list(zones_df[Zones.name])

    Tg: dict[GeneratorId, list[TrancheId]] = {g: [] for g in G}
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

    tranche_cost: dict[OfferId, MoneyUSDPerMW] = {
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
        sum_(tranche_cost[g, t] * p[g, t] for g, t in tranche_cost.keys()),
        grb.GRB.MINIMIZE,
    )

    # Constraints
    power_flow_constraints: grb.tupledict = model.addConstrs(
        (
            sum_(p[g, t] for g in Gz[z] for t in Tg[g])
            + sum_(w[le] for le in Le.get(z, []))
            - sum_(w[lo] for lo in Lo.get(z, []))
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

    line_power: Series[PowerMW] = pd.Series({l_: w[l_].x for l_ in L})
    line_power.index.name = Lines.name

    # Dispatched output for each generator over its tranches
    dispatched = pd.DataFrame.from_records(
        [
            {
                Offers.generator: g,
                Offers.tranche: t,
                Offers.quantity: pgt.x,
            }
            for (g, t), pgt in p.items()
        ],
    ).set_index(OFFERS_INDEX_LABELS)

    # Sanity check
    total_load: PowerMW = inputs.zones[ZonesOutput.load].sum()
    total_generation: PowerMW = dispatched[Offers.quantity].sum()
    mismatch: PowerMW = total_generation - total_load
    assert abs(mismatch) / total_load < 1e-6

    return status, Solution(
        lines=DataFrame[LinesSolution](
            {
                LinesSolution.quantity: line_power,
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
