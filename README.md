# Mini-ISO 

Simulates electricity market auctions for classroom use.

## Reminders

```bash
pipx install poetry

poetry new mini-iso
cd mini-iso
poetry env use python3.11

poetry add \
  altair gurobipy ipython pandas pandera panel \
  hypothesis pytest
```

## TODO

> It there any benefit in the distinction between offers and generators?

* Data IO:
  - [ ] Implement `from_mpower` using `pandapower`
  - [ ] Implement `to_mpower` using `pandapower`
  - [ ] Implement `test_mpower`

* Generic interface to tabulated data:
  - [ ] Remove selected column
  - [ ] Edit selected column
  - [ ] Add column

* Data validation
  - [ ] Offers don't exceed capacity
  - [ ] User-dependent permissions

---

## Planning

* Time-stamped database abstraction for each table
  
```python
Model = TypeVar("Model") 
TimeStamp: TypeAlias = int

@dataclass(frozen=True, slots=True)
class SharedDataFrame:

  pickle_path: Path
  model: Model

  @classmethod
  def initialize(cls, pickle_path: Path, dataframe: DataFrame[Model]) -> SharedDataFrame[Model]:
    timestamp = 0
    pickle_path = pathlib.Path(pickle_path).normalize()
    with open(pickle_path, "wb") as file:
      pickle.dump(obj=timestamp, protocol=pickle.HIGHEST_PROTOCOL)
      pickle.dump(obj=dataframe, protocol=pickle.HIGHEST_PROTOCOL)
    return cls(pickle_path=pickle_path)

  @property
  def timestamp(self) -> TimeStamp:
    pass

  def get(self) -> tuple[TimeStamp, DataFrame[Model]]:
    pass

  def set(self, dataframe: DataFrame[Model]) -> TimeStamp | None:
    pass

  def update(self, rows: DataFrame[Model]) -> TimeStamp | None:
    pass

```

* Solve every `N` seconds:
  - Load current database
  - Recompute LMPs and dispatch instructions
  - Post-processing
* Admin tab:
  - Load input `.csv` from command line arguments to update/reset database
  - `Solve` button
* Offer tab:
  - `Generator` selection
  - `Offer` table
  - `Submit` & `Refresh` buttons
  - `Total|Offered|Surplus Capacity` fields
* Input tabs:
  - Generators tab:
  - Lines tab:
  - Offers tab:
  - Zones tab:
* Output tabs:
  - Generators tab:
    * 
  - Lines tab:
  - Offers tab:
  - Zones tab:

---

