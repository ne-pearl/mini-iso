import pathlib
from typing import Literal
import pandas as pd
from mini_iso.typing_ import Input


def clean(df: pd.DataFrame, orient: Literal["tight"] = "tight") -> pd.DataFrame:
    return pd.DataFrame.from_dict(df.to_dict(orient=orient), orient=orient)


def load_system(constrained: bool = False) -> Input:
    raw = Input.from_json(pathlib.Path(__file__).parent / "mini_new_england.json")

    """
    DEMAND_FRACTION: Fraction
    MAX_FLOW: PowerMW

    if not constrained:
        # For consistency with Mohsen's original configuration
        DEMAND_FRACTION = 0.1
        MAX_FLOW = 100000000.0
    else:
        DEMAND_FRACTION = 1.0
        MAX_FLOW = 1400.0

    fmax: PowerMW = MAX_FLOW
    raw.lines[Lines.capacity] = +fmax
    raw.zones[Zones.load] *= DEMAND_FRACTION
    """

    # Ensures that, e.g., "Generator_10" comes after "Generator_2"
    raw.generators.sort_index(inplace=True)
    raw.offers.sort_index(inplace=True)

    return Input(
        generators=clean(raw.generators),
        offers=clean(raw.offers),
        lines=clean(raw.lines),
        zones=clean(raw.zones),
    )
