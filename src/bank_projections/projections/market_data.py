import datetime

import pandas as pd
import polars as pl

from bank_projections.utils.combine import Combinable


class CurveData(Combinable):
    def __init__(self, data: pd.DataFrame | None = None):
        if data is None:
            self.data = self._empty_frame()
        else:
            self.data = data.copy()
        self._enforce_schema()

    def combine(self, other: "CurveData") -> "CurveData":
        # Filter out empty frames to avoid pandas FutureWarning about empty/all-NA entries
        frames = [df for df in [self.data, other.data] if not df.empty]
        if not frames:
            combined_data = self._empty_frame()
        elif len(frames) == 1:
            combined_data = frames[0].copy()
        else:
            combined_data = pd.concat(frames, ignore_index=True)
        return CurveData(combined_data)

    def get_curves(self, date: datetime.date) -> "Curves":
        # Find latest date in data before or equal to the given date
        latest_date = self.data[self.data["Date"] <= pd.Timestamp(date)]["Date"].max()
        if pd.isna(latest_date):
            raise ValueError(f"No curve data available for date {date}")
        filtered_data = self.data.loc[self.data["Date"] == latest_date]

        # TODO: Interpolation between dates

        return Curves(filtered_data)

    @staticmethod
    def _empty_frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Date": pd.Series(dtype="datetime64[ns]"),
                "Name": pd.Series(dtype="string"),
                "Type": pd.Series(dtype="string"),
                "Tenor": pd.Series(dtype="string"),
                "Maturity": pd.Series(dtype="string"),
                "Rate": pd.Series(dtype="float64"),
                "MaturityYears": pd.Series(dtype="float64"),
            }
        )

    def _enforce_schema(self) -> None:
        # Ensure all required columns exist
        template = self._empty_frame()
        for col in template.columns:
            if col not in self.data.columns:
                self.data[col] = template[col]

        # Coerce types
        self.data["Date"] = pd.to_datetime(self.data["Date"], errors="coerce")

        for col in ["Name", "Type", "Tenor", "Maturity"]:
            self.data[col] = self.data[col].astype("string").str.strip().str.lower()

        self.data["Rate"] = pd.to_numeric(self.data["Rate"], errors="coerce").astype("float64")

        # (Re)compute MaturityYears from Maturity ensuring float dtype
        self.data["MaturityYears"] = self.data["Maturity"].map(parse_tenor).astype("float64")

        # Column order normalization (optional but keeps consistency)
        self.data = self.data[template.columns]


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


class MarketData(Combinable):
    def __init__(self, curves: CurveData | None = None):
        self.curves = curves or CurveData()

    def combine(self, other: "MarketData") -> "MarketData":
        return MarketData(self.curves.combine(other.curves))

    def get_market_rates(self, date: datetime.date) -> "MarketRates":
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


def parse_tenor(tenor: str) -> float | None:
    if pd.isna(tenor):
        return None
    tenor = tenor.strip().lower()
    num = int("".join(ch for ch in tenor if ch.isdigit()))
    unit = "".join(ch for ch in tenor if ch.isalpha())
    if unit not in TENOR_UNIT_MAP:
        raise ValueError(f"Unknown tenor unit: {tenor}")
    return num * TENOR_UNIT_MAP[unit]
