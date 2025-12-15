import pandas as pd
import polars as pl


class Curves:
    def __init__(self, data: pd.DataFrame | None = None):
        self.data = pd.DataFrame() if data is None else data

    def get_spot_rates(self) -> dict[str, float]:
        spot_rates = self.data.loc[self.data["Type"] == "spot"]

        return dict(zip(spot_rates["Name"] + spot_rates["Tenor"], spot_rates["Rate"], strict=False))

    def get_zero_rates(self) -> pd.DataFrame:
        zero_rates = self.data.loc[self.data["Type"] == "zero", ["Name", "MaturityYears", "Rate"]]

        return zero_rates

    def floating_rate_expr(self) -> pl.Expr:
        return pl.col("ReferenceRate").replace_strict(self.get_spot_rates(), default=pl.lit(None)).cast(pl.Float64)


TENOR_UNIT_MAP = {
    "d": 1 / 365.25,
    "w": 7 / 365.25,
    "m": 1 / 12,
    "y": 1.0,
}


def parse_tenor(tenor: str) -> float | None:
    if pd.isna(tenor):
        return None
    tenor = tenor.strip().lower()
    num = int("".join(ch for ch in tenor if ch.isdigit()))
    unit = "".join(ch for ch in tenor if ch.isalpha())
    if unit not in TENOR_UNIT_MAP:
        raise ValueError(f"Unknown tenor unit: {tenor}")
    return num * TENOR_UNIT_MAP[unit]
