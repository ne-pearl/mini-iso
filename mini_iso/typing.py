from __future__ import annotations
import dataclasses
import json
import pathlib
from typing import Any, Callable, Final, Optional, TypeAlias, TypeVar
import pandas as pd
from pandera import DataFrameModel, Field
from pandera.api.pandas import model_config
from pandera.typing import DataFrame, Index, Series
from pandera.errors import SchemaError


GeneratorId: TypeAlias = str
LineId: TypeAlias = int
TrancheId: TypeAlias = str
ZoneId: TypeAlias = str
OfferId: TypeAlias = tuple[GeneratorId, TrancheId]
Fraction: TypeAlias = float
MassKgPerMW: TypeAlias = float
MoneyUSDPerMW: TypeAlias = float
PowerMW: TypeAlias = float
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
        return DataFrame[Model](result)
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
    cost: Series[MoneyUSDPerMW] = _float_field()
    is_included: Optional[Series[bool]] = Field(default=True)

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
    price: Series[MoneyUSDPerMW] = _float_field()

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
    price_offered: Series[MoneyUSDPerMW] = _float_field()
    price_lmp: Series[MoneyUSDPerMW] = _float_field()
    premium: Series[MoneyUSDPerMW] = _float_field()

class Zones(DataFrameModel):
    """Schema for zone data (aggregating bus data)."""

    class Config(model_config.BaseConfig):
        add_missing_columns: bool = True
        unique_column_names: bool = True

    # Inputs
    name: Index[str] = Field(check_name=True, unique=True)
    load: Series[PowerMW] = _float_field()

    # Calculated
    price: Series[MoneyUSDPerMW] = _float_field()
    x: Series[SpatialCoordinate] = _float_field()
    y: Series[SpatialCoordinate] = _float_field()

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
    name: Index[str] = Field(check_name=True, unique=True)
    price: Series[MoneyUSDPerMW] = _float_field()


@dataclasses.dataclass(frozen=True, slots=True)
class Input:
    generators: DataFrame[Generators]
    offers: DataFrame[Offers]
    lines: DataFrame[Lines]
    zones: DataFrame[Zones]

    # # Positive base power
    # base_power: PowerMW = 1000

    # # Big constant
    # big_m: float = 10000

    def __post_init__(self):
        """Validate inter-table relations."""

        zones_id = set(self.zones.index)
        assert zones_id.issuperset(self.generators.zone)
        assert zones_id.issuperset(self.lines.zone_from)
        assert zones_id.issuperset(self.lines.zone_to)

        generators_id = set(self.generators.index)
        assert generators_id.issuperset(self.offers.index.get_level_values(0))

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
    name: Index[str]


class LinesSolution(DataFrameModel):
    # FIXME: For consistency, Index[int] should be Index[str]
    name: Index[int] = Field(unique=True)
    quantity: Series[PowerMW] = Field(coerce=True)


class OffersSolution(DataFrameModel):
    generator: Index[str]
    tranche: Index[str]
    quantity_dispatched: Series[PowerMW] = Field(coerce=True)


class ZonesSolution(DataFrameModel):
    name: Index[str] = Field(unique=True)
    price: Series[MoneyUSDPerMW] = Field(coerce=True)


@dataclasses.dataclass(frozen=True, slots=True)
class Solution:
    lines: DataFrame[LinesSolution]
    offers: DataFrame[OffersSolution]
    zones: DataFrame[ZonesSolution]


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
    price: Series[MoneyUSDPerMW] = Field(coerce=True)
    quantity: Series[PowerMW] = Field(coerce=True)
    quantity_dispatched: Series[PowerMW] = Field(coerce=True)
    utilization: Series[Fraction] = Field(coerce=True)
    # The "nullable=True" appears to be ignored, perhaps
    # because of this:
    # https://pandera.readthedocs.io/en/stable/dtype_validation.html#how-data-types-interact-with-nullable
    #   "datatypes that are inherently not nullable will
    #    fail even if you specify nullable=True because
    #    pandera considers type checks a first-class check
    #    that’s distinct from any downstream check that
    #    you may want to apply to the data"
    is_marginal: Series[bool] = Field(nullable=True)


class ZonesOutput(ZonesSolution):
    load: Series[PowerMW] = Field(coerce=True)
    capacity: Series[PowerMW] = Field(coerce=True)
    dispatched: Series[PowerMW] = Field(coerce=True)
    utilization: Series[PowerMW] = Field(coerce=True)
    x: Series[SpatialCoordinate] = Field(coerce=True)
    y: Series[SpatialCoordinate] = Field(coerce=True)
