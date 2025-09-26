import polars as pl

from bank_projections.financials.balance_sheet import BalanceSheet, MutationReason
from bank_projections.financials.balance_sheet_item import BalanceSheetItem
from bank_projections.projections.coupon_type import CouponTypeRegistry
from bank_projections.projections.frequency import FrequencyRegistry, interest_accrual
from bank_projections.projections.redemption import RedemptionRegistry
from bank_projections.projections.rule import Rule
from bank_projections.projections.time import TimeIncrement


class Runoff(Rule):
    def apply(self, bs: BalanceSheet, increment: TimeIncrement, market_rates) -> BalanceSheet:
        # Apply runoff to all instruments that have maturity dates # TODO: refine
        item = BalanceSheetItem(expr=pl.col("MaturityDate").is_not_null())

        matured = pl.col("MaturityDate") <= pl.lit(increment.to_date)
        number_of_payments = FrequencyRegistry.number_due(
            pl.col("NextCouponDate"), pl.min_horizontal(pl.col("NextCouponDate"), pl.lit(increment.to_date))
        )
        previous_coupon_date = FrequencyRegistry.previous_coupon_date(increment.to_date)
        new_coupon_date = FrequencyRegistry.next_coupon_date(increment.to_date)
        payments = pl.col("Quantity") * pl.col("InterestRate") * FrequencyRegistry.portion_year() * number_of_payments
        floating_rates = market_rates.curves.floating_rate_expr()
        interest_rates = (
            pl.when(number_of_payments > 0)
            .then(CouponTypeRegistry.coupon_rate(floating_rates))
            .otherwise("InterestRate")
        )

        repayment_factors = (
            pl.when(matured)
            .then(1.0)
            .otherwise(
                RedemptionRegistry.redemption_factor(
                    pl.col("MaturityDate"), pl.col("InterestRate"), pl.col("NextCouponDate"), increment.to_date
                )
            )
        )
        prepayment_factors = pl.col("PrepaymentRate").fill_null(0.0) * increment.portion_year
        redemption_factors = 1 - (1 - repayment_factors) * (1 - prepayment_factors)

        new_quantity = pl.col("Quantity") * (1 - redemption_factors) + pl.when(pl.col("IsAccumulating")).then(
            payments
        ).otherwise(0.0)

        new_impairment = pl.when(matured).then(0.0).otherwise(pl.col("Impairment") * (1 - redemption_factors))

        new_accrual = interest_accrual(
            new_quantity,
            interest_rates,
            FrequencyRegistry.portion_passed(new_coupon_date, increment.to_date),
            FrequencyRegistry.portion_year(),
            pl.col("MaturityDate"),
            increment.to_date,
        )

        # For now, assume agio decreases linear
        new_agio = (
            pl.when(matured)
            .then(0.0)
            .otherwise(
                pl.col("Agio")
                * (pl.col("MaturityDate") - increment.to_date).dt.total_days()
                / (pl.col("MaturityDate") - increment.from_date).dt.total_days()
            )
        )

        bs.mutate(
            item,
            pnls={
                MutationReason(module="Runoff", rule="Interest Income"): new_accrual
                - pl.col("AccruedInterest")
                + payments
                + new_agio
                - pl.col("Agio"),
                MutationReason(module="Runoff", rule="Impairment"): new_impairment - pl.col("Impairment"),
            },
            cashflows={
                MutationReason(module="Runoff", rule="Coupon payment"): pl.when(pl.col("IsAccumulating"))
                .then(0.0)
                .otherwise(payments),
                MutationReason(module="Runoff", rule="Principal Repayment"): pl.col("Quantity") * repayment_factors,
                MutationReason(module="Runoff", rule="Principal Prepayment"): pl.col("Quantity")
                * (1 - repayment_factors)
                * prepayment_factors,
            },
            Quantity=new_quantity,
            AccruedInterest=new_accrual,
            Agio=new_agio,
            Impairment=new_impairment,
            PreviousCouponDate=previous_coupon_date,
            NextCouponDate=new_coupon_date,
            FloatingRate=floating_rates,
            InterestRate=interest_rates,
        )

        return bs
