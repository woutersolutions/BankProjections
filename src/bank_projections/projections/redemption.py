import polars as pl

from bank_projections.financials.balance_sheet import BalanceSheet, MutationReason
from bank_projections.financials.balance_sheet_category import BalanceSheetCategoryRegistry
from bank_projections.financials.balance_sheet_item import BalanceSheetItem
from bank_projections.financials.market_data import MarketRates
from bank_projections.projections.redemption_type import RedemptionTypeRegistry
from bank_projections.projections.rule import Rule
from bank_projections.utils.time import TimeIncrement


class Redemption(Rule):
    def apply(self, bs: BalanceSheet, increment: TimeIncrement, market_rates: MarketRates) -> BalanceSheet:
        if increment.from_date == increment.to_date:  # No time passed
            return bs

        matured = pl.col("MaturityDate") <= pl.lit(increment.to_date)

        repayment_factors = (
            pl.when(matured)
            .then(1.0)
            .otherwise(
                RedemptionTypeRegistry.redemption_factor(
                    pl.col("MaturityDate"), pl.col("InterestRate"), pl.col("NextCouponDate"), increment.to_date
                )
            )
        )
        prepayment_factors = pl.col("PrepaymentRate").fill_null(0.0) * increment.portion_year
        redemption_factors = 1 - (1 - repayment_factors) * (1 - prepayment_factors)

        new_nominal = pl.col("Nominal") * (1 - redemption_factors)

        new_impairment = pl.when(matured).then(0.0).otherwise(pl.col("Impairment") * (1 - redemption_factors))

        # Also, assume the valuation error decreases linear
        new_valuation_error = (
            pl.when(pl.col("MaturityDate").is_null())
            .then(pl.col("ValuationError"))
            .when(matured)
            .then(0.0)
            .otherwise(
                pl.col("ValuationError")
                * (pl.col("MaturityDate") - increment.to_date).dt.total_days()
                / (pl.col("MaturityDate") - increment.from_date).dt.total_days()
            )
        )

        signs = BalanceSheetCategoryRegistry.book_value_sign()
        repayment = pl.col("Nominal") * repayment_factors
        prepayment = pl.col("Nominal") * (1 - repayment_factors) * prepayment_factors
        impairment_change = new_impairment - pl.col("Impairment")

        bs.mutate(
            BalanceSheetItem(),
            pnls={
                MutationReason(module="Runoff", rule="Impairment"): signs * impairment_change,
            },
            cashflows={
                MutationReason(module="Runoff", rule="Principal Repayment"): signs * repayment,
                MutationReason(module="Runoff", rule="Principal Prepayment"): signs * prepayment,
            },
            Nominal=new_nominal,
            Impairment=new_impairment,
            ValuationError=new_valuation_error,
            Repayment=repayment,
            Prepayment=prepayment,
            ImpairmentChange=impairment_change,
        )

        return bs
