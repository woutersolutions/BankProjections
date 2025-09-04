import polars as pl

from bank_projections.financials.balance_sheet import BalanceSheet, BalanceSheetItem, MutationReason
from bank_projections.projections.frequency import FrequencyRegistry
from bank_projections.projections.rule import Rule, TimeIncrement


class CouponPayments(Rule):
    def apply(self, bs: BalanceSheet, increment: TimeIncrement) -> BalanceSheet:
        # Polars Series indicating which loans have coupon payments due in the period

        number_of_payments = FrequencyRegistry.number_due(
            pl.col("NextCouponDate"), pl.min_horizontal(pl.col("NextCouponDate"), pl.lit(increment.to_date))
        )
        new_coupon_date = FrequencyRegistry.advance_next(pl.col("NextCouponDate"), number_of_payments)
        payments = pl.col("Quantity") * pl.col("InterestRate") * FrequencyRegistry.portion_year() * number_of_payments
        new_accrual = (
            pl.when(pl.col("MaturityDate") > pl.lit(increment.to_date))
            .then(
                pl.col("Quantity")
                * pl.col("InterestRate")
                * FrequencyRegistry.portion_year()
                * FrequencyRegistry.portion_passed(new_coupon_date, increment.to_date)
            )
            .otherwise(0.0)
        )

        item = BalanceSheetItem()
        reason = MutationReason(module="Runoff", rule="Interest Income")

        bs.mutate(
            item,
            reason,
            Quantity=pl.col("Quantity") + pl.when(pl.col("IsAccumulating")).then(payments).otherwise(0.0),
            AccruedInterest=new_accrual,
            pnl=new_accrual - pl.col("AccruedInterest") + payments,
            liquidity=pl.when(pl.col("IsAccumulating")).then(0.0).otherwise(payments),
        )

        return bs
