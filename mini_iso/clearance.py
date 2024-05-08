from __future__ import annotations
import dataclasses
import enum
import math
import sys
from typing import Final
import gurobipy as grb
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pandera.typing import DataFrame, Series
import scipy
from mini_iso.typing_ import (
    OFFERS_INDEX_LABELS,
    Input,
    Generators,
    GeneratorId,
    LineId,
    Lines,
    LinesSolution,
    PriceUSDPerMWh,
    OfferId,
    Offers,
    OffersSolution,
    PowerMW,
    Solution,
    SusceptanceS,
    TrancheId,
    ZoneId,
    Zones,
    ZonesOutput,
    ZonesSolution,
)


class BasisStatus(enum.Enum):
    BASIC = grb.GRB.BASIC
    NONBASIC_LOWER = grb.GRB.NONBASIC_LOWER
    NONBASIC_UPPER = grb.GRB.NONBASIC_UPPER
    SUPERBASIC = grb.GRB.SUPERBASIC

    @classmethod
    def from_int(cls, status: int) -> BasisStatus | None:
        for e in cls:
            if e.value == status:
                return e


class Status(enum.Enum):
    INFEASIBLE = "infeasible problem"
    OPTIMAL = "optimal solution"
    UNBOUNDED = "unbounded solution"
    UNKNOWN = "uknown or unsupported"


@dataclasses.dataclass(frozen=True, slots=True)
class VariableDuals:

    lb_coef: Series[float]
    ub_coef: Series[float]
    lb_rhs: Series[float]
    ub_rhs: Series[float]
    basis: Series[BasisStatus]

    @classmethod
    def from_variables(cls, variables: grb.tupledict, prefix: str) -> VariableDuals:
        """Initialize from variable bounds constraints."""

        basis_status: NDArray[np.int8] = np.fromiter(
            (v.VBasis for v in variables.values()), dtype=np.int8
        )
        reduced_costs: NDArray[np.float64] = np.fromiter(
            (v.RC for v in variables.values()), dtype=np.float64
        )

        index = pd.Index(data=variables.keys(), tupleize_cols=True)

        def series(data, suffix: str, index=index, prefix=prefix):
            return Series(data=data, index=index, name=f"{prefix}_{suffix}")

        return cls(
            lb_coef=series(
                np.where(basis_status == grb.GRB.NONBASIC_LOWER, reduced_costs, 0.0),
                suffix="lb_coef",
            ),
            ub_coef=series(
                np.where(basis_status == grb.GRB.NONBASIC_UPPER, reduced_costs, 0.0),
                suffix="ub_coef",
            ),
            lb_rhs=series(
                np.fromiter((v.lb for v in variables.values()), dtype=np.float64),
                suffix="lb_rhs",
            ),
            ub_rhs=series(
                np.fromiter((v.ub for v in variables.values()), dtype=np.float64),
                suffix="ub_rhs",
            ),
            basis=series(
                (BasisStatus.from_int(s) for s in basis_status),
                suffix="basis",
            ),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class EqualityDuals:

    coef: Series[float]
    rhs: Series[float]

    @classmethod
    def init(cls, constraints: grb.tupledict, prefix: str) -> EqualityDuals:

        index = pd.Index(data=constraints.keys(), tupleize_cols=True)

        def series(data, suffix: str, index=index, prefix=prefix):
            return Series(data=data, index=index, name=f"{prefix}_{suffix}")

        return cls(
            coef=series((c.Pi for c in constraints.values()), suffix="coef"),
            rhs=series((c.RHS for c in constraints.values()), suffix="rhs"),
        )


def _duality_gap(model: grb.Model) -> float:
    """
    Thanks to
    https://support.gurobi.com/hc/en-us/community/posts/6742325654929-Issue-with-dual-values-for-continuous-model
    """

    total: float = model.ObjCon

    for c in model.getConstrs():
        total += c.RHS * c.Pi

    for v in model.getVars():
        if v.VBasis == grb.GRB.NONBASIC_LOWER:
            total += v.RC * v.lb
        if v.VBasis == grb.GRB.NONBASIC_UPPER:
            total += v.RC * v.ub

    assert model.ModelSense is grb.GRB.MINIMIZE
    gap: float = model.ObjVal - total
    print(f"duality gap: {gap:.2e} ({gap / model.ObjVal * 100:.3f}%)")

    return gap


def clear_auction(inputs: Input) -> tuple[Status, Solution | None]:
    """Solves linear pricing problem for real-time market."""

    base_power: float = inputs.base_power

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

    B: dict[LineId, SusceptanceS] = dict(
        zip(lines_df[Lines.name], lines_df[Lines.susceptance])
    )

    # Parameters:
    tranche_pmax: dict[OfferId, PowerMW] = {
        (str(g), str(t)): p_max
        for g, group in offers_df.groupby(Offers.generator)
        for t, p_max in zip(group[Offers.tranche], group[Offers.quantity])
    }  # max output for generator offers

    # FIXME: Use these updated bounds in place of extra constraints
    # tranche_pmax.update({(g, t): 0.0 for g in GP for t in Tg[g]})  # excluded units

    tranche_cost: dict[OfferId, PriceUSDPerMWh] = {
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
    zones_balance_constraints: grb.tupledict = model.addConstrs(
        (
            sum_(p[g, t] for g in Gz[z] for t in Tg[g])
            + sum_(w[le] for le in Le.get(z, []))
            - sum_(w[lo] for lo in Lo.get(z, []))
            == zone_load[z]
            for z in Z
        ),
        name="power_flow",
    )
    lines_angle_constraints: grb.tupledict = model.addConstrs(
        (
            base_power * B[l_] * (theta[zone_pair[0]] - theta[zone_pair[1]]) == w[l_]
            for l_ in L
            if (zone_pair := Czl[l_]) or exit("unreachable")
        ),
        name="voltage_angle",
    )
    reference_angle_constraint: grb.Constr | None = None
    if len(Czl) != 0:
        first_pair: tuple[ZoneId, ZoneId] = list(Czl.values())[0]
        reference_zone: ZoneId = first_pair[0]
        reference_angle_constraint = model.addConstr(
            theta[reference_zone] == 0.0,
            name="angle_ref",
        )

    # FIXME: Replace with zero bounds on excluded generators
    assert len(GP) == 0
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
        print(f"... succeeded: optimal objective {model.ObjVal:g}")
    else:
        print(f"... failed with status {model.Status} ({status.value})")
        return status, None

    assert status is Status.OPTIMAL
    assert model.ObjCon == 0.0
    assert abs(_duality_gap(model)) < 1e-12 * model.ObjVal

    price = pd.Series(
        {z: constr.Pi for z, constr in zones_balance_constraints.items()},
        name=ZonesSolution.price,
    )
    price.index.name = Zones.name

    angle = pd.Series(
        {z: theta[z].x for z in Z},
        name=ZonesSolution.angle,
    )
    angle.index.name = Zones.name

    line_power: Series[PowerMW] = pd.Series(
        data={l_: w[l_].x for l_ in L},
        name=LinesSolution.quantity,
    )
    line_power.index.name = Lines.name

    # Dispatched output for each generator over its tranches
    dispatched = pd.DataFrame.from_records(
        [
            {
                OffersSolution.generator: g,
                OffersSolution.tranche: t,
                OffersSolution.quantity_dispatched: pgt.x,
            }
            for (g, t), pgt in p.items()
        ],
    ).set_index(OFFERS_INDEX_LABELS)[OffersSolution.quantity_dispatched]

    # Sanity check
    total_load: PowerMW = inputs.zones[ZonesOutput.load].sum()
    total_generation: PowerMW = dispatched.sum()
    mismatch: PowerMW = total_generation - total_load
    assert abs(mismatch) / total_load < 1e-6

    def align(left: DataFrame, right: Series) -> Series:
        merged = left[[]].merge(right, how="left", left_index=True, right_index=True)
        assert all(left.index.values == merged.index.values)
        return merged[right.name]

    lines_quantity: Series = align(inputs.lines, line_power)
    zones_price_solution: Series = align(inputs.zones, price)
    zones_angle_solution: Series = align(inputs.zones, angle)
    offers_solution: Series = align(inputs.offers, dispatched)
    offers_offered_price = Series(
        inputs.offers.price,
        name=OffersSolution.offered_price,
    )

    zones_angles_dual = VariableDuals.from_variables(theta, prefix="angle")
    lines_quantity_dual = VariableDuals.from_variables(w, prefix="quantity")
    offers_quantity_dual = VariableDuals.from_variables(p, prefix="quantity")

    power_flow_duals = EqualityDuals.init(
        zones_balance_constraints,
        prefix="balance",
    )
    angle_duals = EqualityDuals.init(
        lines_angle_constraints,
        prefix="angle",
    )

    num_lines: Final[int] = lines_df.index.size
    num_offers: Final[int] = offers_df.index.size
    num_zones: Final[int] = zones_df.index.size

    lines_from: Series[ZoneId] = lines_df[Lines.zone_from]
    lines_to: Series[ZoneId] = lines_df[Lines.zone_to]
    offers_zone: Series[ZoneId] = pd.merge(
        left=offers_df[Offers.generator],
        right=generators_df[[Generators.name, Generators.zone]],
        how="left",
        left_on=Offers.generator,
        right_on=Generators.name,
    )[Generators.zone]

    zones_index = Series[int](
        data=zones_df.index.values,
        index=zones_df[Zones.name],
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
                        lines_df.index.values,  # out
                        lines_df.index.values,  # on
                    )
                ),
            ),
        ),
        shape=(num_zones, num_lines),
    )

    zones_offers_incidence = scipy.sparse.csc_array(
        (
            np.ones(offers_df.index.size),
            (
                zones_index.loc[offers_zone].values,
                offers_df.index.values,
            ),
        ),
        shape=(num_zones, num_offers),
    )

    return status, Solution(
        objective=model.ObjVal,
        zones_lines_incidence=zones_lines_incidence,
        zones_offers_incidence=zones_offers_incidence,
        base_power=base_power,
        reference_angle_coef=reference_angle_constraint.Pi,
        reference_angle_rhs=0.0,
        lines=DataFrame[LinesSolution](
            pd.concat(
                (
                    lines_quantity,
                    lines_quantity_dual.lb_coef,
                    lines_quantity_dual.lb_rhs,
                    lines_quantity_dual.ub_coef,
                    lines_quantity_dual.ub_rhs,
                    lines_quantity_dual.basis,
                    angle_duals.coef,
                    angle_duals.rhs,
                ),
                axis="columns",
            ),
            index=lines_quantity.index,
        ),
        offers=DataFrame[OffersSolution](
            pd.concat(
                (
                    offers_solution,
                    offers_offered_price,
                    offers_quantity_dual.lb_coef,
                    offers_quantity_dual.lb_rhs,
                    offers_quantity_dual.ub_coef,
                    offers_quantity_dual.ub_rhs,
                    offers_quantity_dual.basis,
                ),
                axis="columns",
            ),
            index=offers_solution.index,
        ),
        zones=DataFrame[ZonesSolution](
            pd.concat(
                (
                    zones_price_solution,
                    zones_angle_solution,
                    zones_angles_dual.lb_coef,
                    zones_angles_dual.lb_rhs,
                    zones_angles_dual.ub_coef,
                    zones_angles_dual.ub_rhs,
                    zones_angles_dual.basis,
                    power_flow_duals.coef,
                    power_flow_duals.rhs,
                ),
                axis="columns",
            ),
            index=zones_price_solution.index,
        ),
    )
