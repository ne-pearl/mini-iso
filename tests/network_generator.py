import collections
import dataclasses
import numbers
from typing import Iterator, Final, Sequence, TypeVar
import warnings
from hypothesis.errors import NonInteractiveExampleWarning
from hypothesis.extra import pandas as hppd
from hypothesis import strategies as hpst
import networkx as nx
import numpy as np
import pandas as pd
from pandera.typing import DataFrame
from mini_iso.typing import (
    OFFERS_INDEX_LABELS,
    Fraction,
    GeneratorId,
    Generators,
    Input,
    LineId,
    Lines,
    PriceUSDPerMWh,
    Offers,
    PowerMW,
    Susceptance,
    ZoneId,
    Zones,
)

T = TypeVar("T")


@dataclasses.dataclass(frozen=True, slots=True)
class ClosedRange(collections.abc.Mapping[str, T]):
    min_value: T
    max_value: T

    def __iter__(self) -> Iterator[str]:
        return iter(f.name for f in dataclasses.fields(type(self)))

    def __len__(self) -> int:
        return 2

    def __getitem__(self, item: str) -> T:
        return getattr(self, item)


GENERATORS_NUM: Final = ClosedRange[int](2, 10)
GENERATORS_CAPACITY_MW: Final = ClosedRange[PowerMW](200.0, 2000.0)
GENERATORS_COST_USD_PER_MW: Final = ClosedRange[PriceUSDPerMWh](0.0, 500.0)

LINES_COTREE_DENSITY: Final = ClosedRange[Fraction](0.3, 1.0)
LINES_CAPACITY_MW: Final = ClosedRange[PowerMW](10, 1000)
LINES_SUSCEPTANCE_S: Final = ClosedRange[Susceptance](15, 150)

OFFERS_NUM_PER_GENERATOR: Final = ClosedRange[int](1, 4)

ZONES_NUM: Final = ClosedRange[int](1, 10)
ZONES_LOAD_MW: Final = ClosedRange[PowerMW](0.5e3, 1e5)

FINITE: Final[dict[str, bool]] = dict(
    allow_infinity=False,
    allow_nan=False,
    allow_subnormal=False,
)


def indexed_names(prefix: str, size: int) -> list[str]:
    return [f"{prefix}{1 + id}" for id in range(size)]


default_zones_size = hpst.integers(**ZONES_NUM)
default_zones_loads = hpst.integers(**ZONES_LOAD_MW)


@hpst.composite
def zones_dataframe(
    draw: hpst.DrawFn,
    size=default_zones_size,
    loads=default_zones_loads,
) -> DataFrame[Zones]:
    """Dataframe of zone/node parameters."""
    num_zones: int = draw(size)
    index = pd.Index(
        data=indexed_names(prefix="Z", size=num_zones),
        dtype=ZoneId,
        name=Zones.name,
    )
    return draw(
        hppd.data_frames(
            columns=[hppd.column(name=Zones.load, dtype=PowerMW, elements=loads)],
            index=hpst.just(index),
        )
    )


default_generators_size = hpst.integers(**GENERATORS_NUM)
default_generators_capacities = hpst.integers(**GENERATORS_CAPACITY_MW)
default_generators_costs = hpst.integers(**GENERATORS_COST_USD_PER_MW)


@hpst.composite
def generators_dataframe(
    draw: hpst.DrawFn,
    zones: Sequence[ZoneId],
    size=default_generators_size,
    capacities=default_generators_capacities,
    costs=default_generators_costs,
) -> DataFrame[Generators]:
    """Dataframe of generator parameters."""
    assert ZoneId is str
    num_generators: int = draw(size)
    index = pd.Index(
        data=indexed_names(prefix="G", size=num_generators),
        dtype=GeneratorId,
        name=Generators.name,
    )
    # This is a replacement for random.choices
    zone_indices: list[int] = draw(
        hpst.lists(
            elements=hpst.integers(
                min_value=0,
                max_value=len(zones) - 1,
            ),
            min_size=num_generators,
            max_size=num_generators,
            unique=False,
        )
    )
    df1: pd.DataFrame = pd.Series(
        # Random sampling with replacement; >=0 generators per node
        # data=rng.choices(population=zones, k=num_generators),
        data=zones[zone_indices],
        dtype=ZoneId,
        index=index,
        name=Generators.zone,
    ).to_frame()
    df2: pd.DataFrame = draw(
        hppd.data_frames(
            columns=[
                hppd.column(
                    name=Generators.capacity,
                    dtype=PowerMW,
                    elements=capacities,
                ),
                hppd.column(
                    name=Generators.cost,
                    dtype=PriceUSDPerMWh,
                    elements=costs,
                ),
            ],
            index=hpst.just(index),
        )
    )
    return pd.concat([df1, df2], axis="columns")


default_lines_cotree_density = hpst.floats(**LINES_COTREE_DENSITY)
default_lines_capacities = hpst.integers(**LINES_CAPACITY_MW)
default_lines_susceptances = hpst.integers(**LINES_SUSCEPTANCE_S)


@hpst.composite
def lines_dataframe(
    draw: hpst.DrawFn,
    zones: Sequence[ZoneId],
    cotree_density=default_lines_cotree_density,
    capacities=default_lines_capacities,
    susceptances=default_lines_susceptances,
) -> DataFrame[Lines]:
    """Dataframe of line parameters."""
    # rng: random.Random = hpst.randoms()
    num_zones: int = len(zones)
    graph_complete: nx.Graph = nx.complete_graph(n=num_zones)
    graph_tree: nx.Graph = nx.minimum_spanning_tree(graph_complete)
    graph_cotree: nx.Graph = nx.difference(graph_complete, graph_tree)
    num_edges: int = round(graph_cotree.size() * draw(cotree_density))
    # Random sampling without replacement to prevent duplicates
    # cotree_edge_sample: list[tuple[int, int]] = rng.sample(
    #     list(graph_cotree.edges()),
    #     k=num_edges,
    # )
    # FIXME: This is a replacement for random.sample, but seems far less efficient.
    # Hypothesis warns us "HypothesisDeprecationWarning: Do not use the `random` module
    # inside strategies; instead consider  `st.randoms()`, `st.sampled_from()`, etc."
    cotree_edge_indices = draw(
        hpst.lists(
            elements=hpst.integers(
                min_value=0,
                max_value=max(graph_cotree.number_of_edges() - 1, 0),
            ),
            min_size=num_edges,
            max_size=num_edges,
            unique=True,
        )
    )
    cotree_edges: list[tuple[int, int]] = list(graph_cotree.edges())
    cotree_edge_sample = [cotree_edges[i] for i in cotree_edge_indices]
    graph_cotree_subgraph: nx.Graph = nx.from_edgelist(cotree_edge_sample)
    graph_sampled: nx.Graph = nx.compose(graph_tree, graph_cotree_subgraph)
    edges: list[tuple[ZoneId, ZoneId]] = sorted(graph_sampled.edges())
    num_edges: int = len(edges)
    index_from: tuple[int] = ()
    index_to: tuple[int] = ()
    if len(edges) != 0:
        index_from, index_to = zip(*sorted(edges))
    zone_from: list[ZoneId] = [zones[id] for id in index_from]
    zone_to: list[ZoneId] = [zones[id] for id in index_to]
    index = pd.Index(
        data=[1 + id for id in range(num_edges)],
        dtype=LineId,
        name=Lines.name,
    )

    size: dict[str, int] = dict(min_size=num_edges, max_size=num_edges)
    capacities_draw = draw(hpst.lists(elements=capacities, **size))
    susceptances_draw = draw(hpst.lists(elements=susceptances, **size))

    return DataFrame[Lines](
        {
            Lines.capacity: capacities_draw,
            Lines.susceptance: susceptances_draw,
            Lines.zone_from: zone_from,
            Lines.zone_to: zone_to,
        },
        index=index,
    )


default_offers_num_per_generator = hpst.integers(**OFFERS_NUM_PER_GENERATOR)


@hpst.composite
def offers_dataframe(
    draw: hpst.DrawFn,
    names: Sequence[GeneratorId],
    costs: Sequence[PriceUSDPerMWh],
    capacities: Sequence[PowerMW],
    num_offers_per_generator=default_offers_num_per_generator,
):
    """Offer data."""

    num_generators = len(names)
    assert len(costs) == num_generators
    assert len(capacities) == num_generators

    num_offers_per_generator_draw = draw(
        hpst.lists(
            elements=num_offers_per_generator,
            min_size=num_generators,
            max_size=num_generators,
        )
    )

    def make_generator_offers(
        name: str, capacity: PowerMW, cost: PriceUSDPerMWh, num_offers: int
    ) -> DataFrame[Offers]:
        def normalize(elements: list[numbers.Number]) -> list[float]:
            total: numbers.Number = sum(elements)
            assert total > 0.0
            return [e / total for e in elements]

        size = dict(min_size=num_offers, max_size=num_offers)
        capacity_fractions: list[float] = normalize(
            draw(
                hpst.lists(
                    elements=hpst.integers(min_value=1, max_value=max(1, num_offers)),
                    **size,
                )
            )
        )
        cost_fractions: list[float] = draw(
            hpst.lists(
                elements=hpst.floats(min_value=0.8, max_value=2.0),
                **size,
            )
        )

        multi_index = pd.MultiIndex.from_arrays(
            arrays=(
                [name] * num_offers,  # generator
                indexed_names(prefix="T", size=num_offers),  # tranches
            ),
            names=OFFERS_INDEX_LABELS,
        )

        return DataFrame[Offers](
            {
                Offers.quantity: np.array(capacity_fractions) * capacity,
                Offers.price: np.array(cost_fractions) * cost,
            },
            index=multi_index,
        )

    dataframes: list[DataFrame[Offers]] = [
        make_generator_offers(
            name=name,
            capacity=capacity,
            cost=cost,
            num_offers=num_offers,
        )
        for name, capacity, cost, num_offers in zip(
            names, capacities, costs, num_offers_per_generator_draw
        )
    ]
    return pd.concat(dataframes, axis="rows")


@hpst.composite
def networks(
    draw: hpst.DrawFn,
    # Generator parameters
    generators_size=default_generators_size,
    generators_capacities=default_generators_capacities,
    generators_costs=default_generators_costs,
    # Lines parameters
    lines_cotree_density=default_lines_cotree_density,
    lines_capacities=default_lines_capacities,
    lines_susceptances=default_lines_susceptances,
    # Offer parameters
    offers_num_per_generator=default_offers_num_per_generator,
    offers_safety_factor: float = 10.0,
    # Zone parameters
    zones_size=default_zones_size,
    zones_loads=default_zones_loads,
) -> Input:
    zones: DataFrame[Zones] = draw(
        zones_dataframe(
            size=zones_size,
            loads=zones_loads,
        )
    )

    generators: DataFrame[Generators] = draw(
        generators_dataframe(
            zones=zones.index.values,
            size=generators_size,
            capacities=generators_capacities,
            costs=generators_costs,
        )
    )

    lines: DataFrame[Lines] = draw(
        lines_dataframe(
            zones=zones.index.values,
            cotree_density=lines_cotree_density,
            capacities=lines_capacities,
            susceptances=lines_susceptances,
        )
    )

    offers: DataFrame[Offers] = draw(
        offers_dataframe(
            names=generators.index.values,
            costs=generators[Generators.cost].values,
            capacities=generators[Generators.capacity].values,
            num_offers_per_generator=offers_num_per_generator,
        )
    )

    ratio = zones[Zones.load].sum() / offers[Offers.quantity].sum()
    offers[Offers.quantity] *= ratio * offers_safety_factor

    return Input(
        generators=generators,
        lines=lines,
        offers=offers,
        zones=zones,
    )


if __name__ == "__main__":
    from mini_iso.clearance import Solution, clear_auction

    warnings.simplefilter(action="ignore", category=FutureWarning)
    warnings.simplefilter(action="ignore", category=NonInteractiveExampleWarning)
    inputs: Input = networks(
        generators_size=hpst.just(8),
        lines_cotree_density=hpst.just(0.8),
        zones_size=hpst.just(4),
    ).example()
    print(inputs.generators)
    print(inputs.offers)
    print(inputs.lines)
    print(inputs.zones)

    print(f"capacity: {inputs.offers[Offers.quantity].sum():.1f} MW")
    print(f"    load: {inputs.zones[Zones.load].sum():.1f} MW")

    solution: Solution = clear_auction(inputs)
