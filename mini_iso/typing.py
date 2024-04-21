from __future__ import annotations
import dataclasses
import enum
import itertools
import json
import pathlib
from typing import Any, Callable, Final, Optional, TypeAlias, TypeVar
import networkx as nx
import numpy as np
import pandas as pd
from pandera import DataFrameModel, Field
from pandera.api.pandas import model_config
from pandera.typing import DataFrame, Index, Series
from pandera.errors import SchemaError

AngleDegrees: TypeAlias = float
GeneratorId: TypeAlias = str
LineId: TypeAlias = str
TrancheId: TypeAlias = str
ZoneId: TypeAlias = str
OfferId: TypeAlias = tuple[GeneratorId, TrancheId]
Fraction: TypeAlias = float
MassKgPerMW: TypeAlias = float
PriceUSDPerMWh: TypeAlias = float
PowerMW: TypeAlias = float
PaymentUSDPerH: TypeAlias = float
SpatialCoordinate: TypeAlias = float
Susceptance: TypeAlias = float

Model = TypeVar("Model")


def _float_field(*args, **kwargs):
    return Field(*args, **kwargs, coerce=True, ignore_na=False)


def _validate_and_reindex(
    model: type[DataFrameModel], df: pd.DataFrame, *index: str
) -> DataFrame[Model]:
    """Validate data against model."""
    result = df.copy(deep=True)
    index_names: tuple[str, ...] = (
        tuple(df.index.names) if df.index.name is None else (df.index.name,)
    )
    if index_names == (None,):
        # Set index if none has explicitly been specified
        # Warning: DataFrame.set_index needs a list; a tuple doesn't work!
        result.set_index(list(index), inplace=True)
    elif index_names != index:
        # Don't attempt to handle a mismatch
        raise ValueError("Index conflict", index_names, index)
    try:
        # FIXME: This doesn't seem right!
        model.validate(result)
        return DataFrame[model](result)
    except SchemaError as error:
        print(f"Schema errors and failure cases for {model}:")
        print(error.failure_cases)
        # The stack trace is very long! Abort instead.
        assert False, "Aborting!"


class Generators(DataFrameModel):
    """Schema for generator data."""

    class Config(model_config.BaseConfig):
        add_missing_columns: bool = True
        unique_column_names: bool = True

    # Inputs
    name: Index[GeneratorId] = Field(check_name=True, unique=True)
    capacity: Series[PowerMW] = _float_field()
    zone: Series[ZoneId]
    cost: Series[PriceUSDPerMWh] = _float_field()
    is_included: Optional[Series[bool]] = Field(default=True)

    x: Optional[Series[SpatialCoordinate]] = _float_field(nullable=True)
    y: Optional[Series[SpatialCoordinate]] = _float_field(nullable=True)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        return _validate_and_reindex(cls, df, cls.name)


class Lines(DataFrameModel):
    """Schema for line data."""

    class Config(model_config.BaseConfig):
        add_missing_columns: bool = True
        unique_column_names: bool = True

    # Inputs
    name: Index[LineId] = Field(check_name=True, coerce=True)  # coerce 3 to "3", etc.
    capacity: Series[PowerMW] = _float_field()
    susceptance: Series[Susceptance] = _float_field()
    zone_from: Series[ZoneId]
    zone_to: Series[ZoneId]

    # Calculated
    # power: Series[PowerMW] = _float_field()
    # slack: Series[PowerMW] = _float_field()

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        return _validate_and_reindex(cls, df, cls.name)


class Offers(DataFrameModel):
    """Schema for offer data."""

    class Config(model_config.BaseConfig):
        multiindex_name = "offer"
        multiindex_strict = True
        unique_column_names: bool = True

    generator: Index[GeneratorId]
    tranche: Index[TrancheId]
    quantity: Series[PowerMW] = _float_field()
    price: Series[PriceUSDPerMWh] = _float_field()

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        return _validate_and_reindex(cls, df, cls.generator, cls.tranche)


OFFERS_INDEX_LABELS: Final[list[str]] = [Offers.generator, Offers.tranche]


class OffersSummary(DataFrameModel):
    class Config(model_config.BaseConfig):
        multiindex_name = "offer"
        multiindex_strict = True
        unique_column_names: bool = True

    generator: Index[GeneratorId]
    tranche: Index[TrancheId]
    quantity_offered: Series[PowerMW] = _float_field()
    quantity_dispatched: Series[PowerMW] = _float_field()
    utilization: Series[Fraction] = _float_field()
    price_offered: Series[PriceUSDPerMWh] = _float_field()
    price_lmp: Series[PriceUSDPerMWh] = _float_field()
    excess: Series[PriceUSDPerMWh] = _float_field()
    revenue: Series[PaymentUSDPerH] = _float_field()


class Zones(DataFrameModel):
    """Schema for zone data (aggregating bus data)."""

    class Config(model_config.BaseConfig):
        add_missing_columns: bool = True
        unique_column_names: bool = True

    # Inputs
    name: Index[ZoneId] = Field(check_name=True, unique=True)
    load: Series[PowerMW] = _float_field()

    x: Optional[Series[SpatialCoordinate]] = _float_field(nullable=True)
    y: Optional[Series[SpatialCoordinate]] = _float_field(nullable=True)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        return _validate_and_reindex(cls, df, cls.name)


class LinesFlow(DataFrameModel):
    name: Index[LineId] = Field(check_name=True, coerce=True)
    quantity: Series[PowerMW] = _float_field()


class OffersDispatched(DataFrameModel):
    class Config(model_config.BaseConfig):
        multiindex_name = "offer"
        multiindex_strict = True
        unique_column_names: bool = True

    generator: Index[GeneratorId]
    tranche: Index[TrancheId]
    quantity_dispatched: Series[PowerMW] = _float_field()


class ZonesPrice(DataFrameModel):
    name: Index[ZoneId] = Field(check_name=True, unique=True)
    price: Series[PriceUSDPerMWh] = _float_field()


class Part(enum.Enum):
    GENERATOR = Generators.__name__
    ZONE = Zones.__name__


Id: TypeAlias = GeneratorId | ZoneId
Node: TypeAlias = tuple[Part, Id]
Pos: TypeAlias = tuple[SpatialCoordinate, SpatialCoordinate]


@dataclasses.dataclass(frozen=True, slots=True)
class Input:
    generators: DataFrame[Generators]
    offers: DataFrame[Offers]
    lines: DataFrame[Lines]
    zones: DataFrame[Zones]

    def __post_init__(self):
        """Validate inter-table relations."""

        zones_id = set(self.zones.index)
        assert zones_id.issuperset(self.generators.zone)
        assert zones_id.issuperset(self.lines.zone_from)
        assert zones_id.issuperset(self.lines.zone_to)

        generators_id = set(self.generators.index)
        assert generators_id.issuperset(self.offers.index.get_level_values(0))

        self._layout_network()

    def _layout_network(self) -> None:
        # See `weight` in networkx.spring_layout()
        # https://networkx.org/documentation/stable/reference/generated/networkx.drawing.layout.spring_layout.html
        line_weight_key: Final[str] = "weight"
        line_weight_zone: Final[int] = 1
        line_weight_generator: Final[int] = line_weight_zone * 10

        zones_graph: nx.Graph = nx.from_edgelist(
            zip(
                ((Part.ZONE, from_) for from_ in self.lines[Lines.zone_from]),
                ((Part.ZONE, to_) for to_ in self.lines[Lines.zone_to]),
                itertools.repeat({line_weight_key: line_weight_zone}),
            ),
            create_using=nx.DiGraph,
        )
        generators_graph: nx.Graph = nx.from_edgelist(
            (
                (
                    (Part.GENERATOR, from_),
                    (Part.ZONE, to_),
                    {line_weight_key: line_weight_generator},
                )
                for from_, to_ in self.generators[Generators.zone].items()
            ),
            create_using=nx.DiGraph,
        )
        graph: nx.Graph = nx.compose(zones_graph, generators_graph)

        def get_pos(
            df: pd.DataFrame, part: Part, x_column: str, y_column: str
        ) -> dict[Node, Pos]:
            """Get positions if available."""
            return (
                {
                    (part, key): xy
                    for key, x, y in zip(df.index, df[x_column], df[y_column])
                    if np.nan not in (xy := (x, y))
                }
                if set(df.columns).issuperset([x_column, y_column])
                else {}
            )

        generators_pos_fixed: dict[Node, Pos] = get_pos(
            df=self.zones,
            part=Part.ZONE,
            x_column=Zones.x,
            y_column=Zones.y,
        )
        zones_pos_fixed: dict[Node, Pos] = get_pos(
            df=self.generators,
            part=Part.GENERATOR,
            x_column=Generators.x,
            y_column=Generators.y,
        )
        pos: dict[Node, Pos] = {**generators_pos_fixed, **zones_pos_fixed}
        fixed: dict[Node, bool] = {node: True for node in pos.keys()}
        layout = nx.spring_layout(
            graph,
            pos=pos or None,  # must be non-empty or None
            fixed=fixed or None,  # must be non-empty or None
            iterations=5000,
            seed=0,
            weight=line_weight_key,
        )

        zones_pos: dict[ZoneId, Pos] = {}
        generators_pos: dict[GeneratorId, Pos] = {}

        node: Node
        pos: Pos
        for node, pos in layout.items():
            if not fixed.get(node, False):
                part: Part
                index: Id
                part, index = node
                reference = generators_pos if part is Part.GENERATOR else zones_pos
                reference[index] = tuple(pos)

        def merge(
            self_: pd.DataFrame, updated_pos: dict[Id, Pos], *columns: Id
        ) -> None:
            for column in set(columns).difference(self_.columns):
                # The column must exist before it can be updated
                self_[column] = None
            self_.update(other=pd.DataFrame(updated_pos, index=columns).T)

        merge(self.generators, generators_pos, Generators.x, Generators.y)
        merge(self.zones, zones_pos, Zones.x, Zones.y)

    @classmethod
    def from_json(cls, path: pathlib.Path | str, **kwargs) -> Input:
        """Load from JSON file."""

        json_path = pathlib.Path(path)
        with open(json_path) as file:
            mapping: dict[str, Any] = json.load(file)

        def load_model(model: type[Model], key: str) -> DataFrame[Model]:
            """Validated load from file."""
            nonlocal mapping
            file_path = pathlib.PurePath(mapping[key])
            loader: Callable[[pathlib.Path], DataFrame] | None = {
                ".csv": pd.read_csv,
            }.get(file_path.suffix)
            if loader is None:
                raise ValueError("Unsupported suffix", file_path, json_path)
            return model.from_dataframe(loader(json_path.parent / file_path))  # type: ignore[attr-defined]

        generators = load_model(Generators, "generators")
        lines = load_model(Lines, "lines")
        offers = load_model(Offers, "offers")
        zones = load_model(Zones, "zones")

        return cls(
            generators=generators,
            lines=lines,
            offers=offers,
            zones=zones,
        )


class GeneratorsSolution(DataFrameModel):
    name: Index[GeneratorId]


class LinesSolution(DataFrameModel):
    name: Index[LineId] = Field(unique=True)
    quantity: Series[PowerMW] = _float_field()


class OffersSolution(DataFrameModel):
    generator: Index[GeneratorId]
    tranche: Index[TrancheId]
    quantity_dispatched: Series[PowerMW] = _float_field()


class ZonesSolution(DataFrameModel):
    name: Index[ZoneId] = Field(unique=True)
    price: Series[PriceUSDPerMWh] = _float_field()


@dataclasses.dataclass(frozen=True, slots=True)
class Solution:
    objective: PaymentUSDPerH
    lines: DataFrame[LinesSolution]
    offers: DataFrame[OffersSolution]
    zones: DataFrame[ZonesSolution]


class GeneratorsOutput(GeneratorsSolution):
    capacity: Series[PowerMW] = _float_field()
    zone: Series[ZoneId]
    dispatched: Series[PowerMW] = _float_field()
    utilization: Series[Fraction] = _float_field()
    nodal_price: Series[PriceUSDPerMWh] = _float_field()
    revenue: Series[PaymentUSDPerH] = _float_field()
    x: Series[SpatialCoordinate] = _float_field()
    y: Series[SpatialCoordinate] = _float_field()
    x_zone: Series[SpatialCoordinate] = _float_field()
    y_zone: Series[SpatialCoordinate] = _float_field()
    x_mid: Series[SpatialCoordinate] = _float_field()
    y_mid: Series[SpatialCoordinate] = _float_field()


class LinesOutput(LinesSolution):
    zone_from: Series[ZoneId]
    zone_to: Series[ZoneId]
    susceptance: Series[Susceptance] = _float_field()
    quantity_abs: Series[PowerMW] = _float_field()
    capacity: Series[PowerMW] = _float_field()
    slack: Series[PowerMW] = _float_field()
    utilization: Series[Fraction]
    is_critical: Series[bool]
    x_from: Series[SpatialCoordinate] = _float_field()
    y_from: Series[SpatialCoordinate] = _float_field()
    x_to: Series[SpatialCoordinate] = _float_field()
    y_to: Series[SpatialCoordinate] = _float_field()
    x_mid: Series[SpatialCoordinate] = _float_field()
    y_mid: Series[SpatialCoordinate] = _float_field()
    angle_degrees: Series[AngleDegrees] = _float_field()

class OffersOutput(DataFrameModel):
    class Config(model_config.BaseConfig):
        multiindex_name = "offer"
        multiindex_strict = True
        unique_column_names: bool = True

    generator: Index[GeneratorId]
    tranche: Index[TrancheId]
    zone: Series[ZoneId]
    quantity: Series[PowerMW] = _float_field()
    quantity_dispatched: Series[PowerMW] = _float_field()
    utilization: Series[Fraction] = _float_field()
    price: Series[PriceUSDPerMWh] = _float_field()
    nodal_price: Series[PriceUSDPerMWh] = _float_field()
    revenue: Series[PaymentUSDPerH] = _float_field()
    # The "nullable=True" appears to be ignored, perhaps
    # because of this:
    # https://pandera.readthedocs.io/en/stable/dtype_validation.html#how-data-types-interact-with-nullable
    #   "datatypes that are inherently not nullable will
    #    fail even if you specify nullable=True because
    #    pandera considers type checks a first-class check
    #    that's distinct from any downstream check that
    #    you may want to apply to the data"
    is_marginal: Series[bool] = Field(nullable=True)


class ZonesOutput(ZonesSolution):
    load: Series[PowerMW] = _float_field()
    capacity: Series[PowerMW] = _float_field()
    dispatched: Series[PowerMW] = _float_field()
    utilization: Series[PowerMW] = _float_field()
    x: Series[SpatialCoordinate] = _float_field()
    y: Series[SpatialCoordinate] = _float_field()
