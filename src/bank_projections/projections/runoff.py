import polars as pl

from bank_projections.financials.balance_sheet import BalanceSheet, MutationReason
from bank_projections.financials.balance_sheet_category import BalanceSheetCategoryRegistry
from bank_projections.financials.balance_sheet_item import BalanceSheetItem
from bank_projections.financials.market_data import MarketRates
from bank_projections.projections.coupon_type import CouponTypeRegistry
from bank_projections.projections.frequency import FrequencyRegistry
from bank_projections.projections.redemption import RedemptionRegistry
from bank_projections.projections.rule import Rule
from bank_projections.utils.coupons import coupon_payment, interest_accrual
from bank_projections.utils.time import TimeIncrement


class Runoff(Rule):
    def apply(self, bs: BalanceSheet, increment: TimeIncrement, market_rates: MarketRates) -> BalanceSheet:
        if increment.from_date == increment.to_date:  # No time passed
            return bs

        matured = pl.col("MaturityDate") <= pl.lit(increment.to_date)
        number_of_payments = FrequencyRegistry.number_due(
            pl.col("NextCouponDate"), pl.min_horizontal(pl.col("MaturityDate"), pl.lit(increment.to_date))
        )
        previous_coupon_date = FrequencyRegistry.previous_coupon_date(increment.to_date)
        new_coupon_date = pl.when(matured).then(None).otherwise(FrequencyRegistry.next_coupon_date(increment.to_date))
        coupon_payments = coupon_payment(pl.col("Nominal"), pl.col("InterestRate")) * number_of_payments
        floating_rates = market_rates.curves.floating_rate_expr()
        new_interest_rates = (
            pl.when(number_of_payments > 0)
            .then(CouponTypeRegistry.coupon_rate(floating_rates))
            .otherwise(pl.col("InterestRate"))
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

        new_nominal = pl.col("Nominal") * (1 - redemption_factors) + pl.when(pl.col("IsAccumulating")).then(
            coupon_payments
        ).otherwise(0.0)

        new_impairment = pl.when(matured).then(0.0).otherwise(pl.col("Impairment") * (1 - redemption_factors))

        new_accrual = interest_accrual(
            new_nominal,
            new_interest_rates,
            previous_coupon_date,
            new_coupon_date,
            increment.to_date,
        ) + pl.col("AccruedInterestError")

        # For now, assume agio decreases linear
        new_agio = (
            pl.when(pl.col("MaturityDate").is_null())
            .then(pl.col("Agio"))
            .when(matured)
            .then(0.0)
            .otherwise(
                pl.col("Agio")
                * (pl.col("MaturityDate") - increment.to_date).dt.total_days()
                / (pl.col("MaturityDate") - increment.from_date).dt.total_days()
            )
        )

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

        bs.mutate(
            BalanceSheetItem(),
            pnls={
                MutationReason(module="Runoff", rule="Accrual"): signs
                * (coupon_payments + (new_accrual - pl.col("AccruedInterest"))),
                MutationReason(module="Runoff", rule="Impairment"): signs * (new_impairment - pl.col("Impairment")),
                MutationReason(module="Runoff", rule="Agio"): signs * (new_agio - pl.col("Agio")),
            },
            cashflows={
                MutationReason(module="Runoff", rule="Coupon payment"): signs
                * pl.when(pl.col("IsAccumulating")).then(0.0).otherwise(coupon_payments),
                MutationReason(module="Runoff", rule="Principal Repayment"): signs
                * pl.when(RedemptionRegistry.has_principal_exchange())
                .then(pl.col("Nominal") * repayment_factors)
                .otherwise(0.0),
                MutationReason(module="Runoff", rule="Principal Prepayment"): signs
                * pl.when(RedemptionRegistry.has_principal_exchange())
                .then(pl.col("Nominal") * (1 - repayment_factors) * prepayment_factors)
                .otherwise(0.0),
            },
            Nominal=new_nominal,
            AccruedInterest=new_accrual,
            Agio=new_agio,
            Impairment=new_impairment,
            PreviousCouponDate=previous_coupon_date,
            NextCouponDate=new_coupon_date,
            FloatingRate=floating_rates,
            InterestRate=new_interest_rates,
            ValuationError=new_valuation_error,
        )

        return bs
