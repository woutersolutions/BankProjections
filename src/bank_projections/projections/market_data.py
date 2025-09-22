import datetime

import pandas as pd
import polars as pl

from bank_projections.utils.combine import Combinable, T


class CurveData(Combinable):
    def __init__(self, data: pd.DataFrame | None = None):
        self.data = (
            pd.DataFrame(columns=["Date", "Name", "Type", "Tenor", "Maturity", "Rate"]) if data is None else data
        )

        for col in ["Name", "Type", "Tenor", "Maturity"]:
            self.data[col] = self.data[col].astype("string").str.strip().str.lower()

        self.data["MaturityYears"] = self.data["Maturity"].map(parse_tenor)

    def combine(self, other: "CurveData") -> "CurveData":
        combined_data = pd.concat([self.data, other.data]).reset_index(drop=True)
        return CurveData(combined_data)

    def get_curves(self, date: datetime.date) -> "Curves":
        # Find latest date in data before or equal to the given date
        latest_date = self.data[self.data["Date"] <= pd.Timestamp(date)]["Date"].max()
        if pd.isna(latest_date):
            raise ValueError(f"No curve data available for date {date}")
        filtered_data = self.data.loc[self.data["Date"] == latest_date]

        # TODO: Interpolation between dates

        return Curves(filtered_data)


class Curves:
    def __init__(self, data: pd.DataFrame | None = None):
        self.data = pd.DataFrame() if data is None else data

    def get_spot_rates(self) -> dict[str, float]:
        spot_rates = self.data.loc[self.data["Type"] == "spot"]

        return dict(zip(spot_rates["Name"] + spot_rates["Tenor"], spot_rates["Rate"], strict=False))

    def floating_rate_expr(self):
        return pl.col("ReferenceRate").replace_strict(self.get_spot_rates(), default=pl.lit(None)).cast(pl.Float64)


class MarketData(Combinable):
    def __init__(self, curves: CurveData | None = None):
        self.curves = curves or CurveData()

    def combine(self, other: "MarketData") -> T:
        return MarketData(self.curves.combine(other.curves))

    def get_market_rates(self, date: datetime.date):
        return MarketRates(self.curves.get_curves(date))


class MarketRates:
    def __init__(self, curves: Curves | None = None):
        self.curves = curves or Curves()


TENOR_UNIT_MAP = {
    "d": 1 / 365.25,
    "w": 7 / 365.25,
    "m": 1 / 12,
    "y": 1.0,
}


def parse_tenor(tenor: str) -> float:
    if pd.isna(tenor):
        return None
    tenor = tenor.strip().lower()
    num = int("".join(ch for ch in tenor if ch.isdigit()))
    unit = "".join(ch for ch in tenor if ch.isalpha())
    if unit not in TENOR_UNIT_MAP:
        raise ValueError(f"Unknown tenor unit: {tenor}")
    return num * TENOR_UNIT_MAP[unit]
