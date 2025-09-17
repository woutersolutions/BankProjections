import datetime

import pandas as pd

from bank_projections.utils.combine import Combinable, T


class CurveData(Combinable):
    def __init__(self, data: pd.DataFrame | None = None):
        self.data = pd.DataFrame(columns=["Date"]) if data is None else data

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
