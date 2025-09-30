import pandas as pd
import polars as pl

from bank_projections.financials.balance_sheet import BalanceSheet, MutationReason
from bank_projections.projections.market_data import MarketRates
from bank_projections.projections.rule import Rule
from bank_projections.projections.time import TimeIncrement
from bank_projections.scenarios.scenario import Scenario
from bank_projections.scenarios.template import ScenarioTemplate


class TaxTemplate(ScenarioTemplate):
    def load_excel_sheet(self, file_path: str, sheet_name: str) -> Scenario:
        df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None).set_index(0)

        # TODO: More comprehensive tax rules
        tax_rate = float(df_raw.iloc[1, 0])

        rule = TaxRule(tax_rate=tax_rate)

        return Scenario(rules={"Tax": rule})


class TaxRule(Rule):
    def __init__(self, tax_rate: float):
        self.tax_rate = tax_rate

    def apply(self, bs: BalanceSheet, increment: TimeIncrement, market_rates: MarketRates) -> BalanceSheet:
        income, expense = bs.pnls.select(
            income=pl.when(pl.col("Amount") < 0).then(pl.col("Amount")).otherwise(0).sum() * -pl.lit(self.tax_rate),
            expense=pl.when(pl.col("Amount") > 0).then(pl.col("Amount")).otherwise(0).sum() * -pl.lit(self.tax_rate),
        ).row(0)

        bs.add_single_pnl(expense, MutationReason(module="Tax", rule="Tax expense"), offset_liquidity=True)
        bs.add_single_pnl(income, MutationReason(module="Tax", rule="Tax benefit"), offset_liquidity=True)

        return bs
