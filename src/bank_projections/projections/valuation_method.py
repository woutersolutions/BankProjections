import datetime
from abc import ABC, abstractmethod

import pandas as pd
import polars as pl

from bank_projections.projections.frequency import FrequencyRegistry
from bank_projections.utils.base_registry import BaseRegistry
from bank_projections.utils.scaling import AdditiveScaling, MultiplicativeScaling, NoScaling, ScalingMethod


class ValuationMethod(ABC):
    @classmethod
    @abstractmethod
    def calculated_dirty_price(
        cls,
        data: pl.DataFrame,
        projection_date: datetime.date,
        zero_rates: pd.DataFrame,
        output_column: str,
    ) -> pl.DataFrame:
        pass

    @classmethod
    def corrected_dirty_price(
        cls,
        data: pl.DataFrame,
        projection_date: datetime.date,
        zero_rates: pd.DataFrame,
        output_column: str,
    ) -> pl.DataFrame:
        return cls.calculated_dirty_price(data, projection_date, zero_rates, output_column).with_columns(
            cls._correction_method().scale(pl.col(output_column), pl.col("ValuationError")).alias(output_column)
        )

    @classmethod
    @abstractmethod
    def _correction_method(cls) -> ScalingMethod:
        pass

    @classmethod
    def valuation_error(cls, calculated_price: pl.Expr, correct_price: pl.Expr) -> pl.Expr:
        return cls._correction_method().scaling_factor(calculated_price, correct_price)


class NoValuationMethod(ValuationMethod):
    @classmethod
    def _correction_method(cls) -> ScalingMethod:
        return NoScaling()

    @classmethod
    def calculated_dirty_price(
        cls,
        data: pl.DataFrame,
        projection_date: datetime.date,
        zero_rates: pd.DataFrame,
        output_column: str,
    ) -> pl.DataFrame:
        return data.with_columns(pl.lit(None, dtype=pl.Float64).alias(output_column))


class AmortizedCostValuationMethod(ValuationMethod):
    @classmethod
    def _correction_method(cls) -> ScalingMethod:
        return MultiplicativeScaling()

    @classmethod
    def calculated_dirty_price(
        cls,
        data: pl.DataFrame,
        projection_date: datetime.date,
        zero_rates: pd.DataFrame,
        output_column: str,
    ) -> pl.DataFrame:
        return data.with_columns(
            (
                1.0
                + pl.when(pl.col("Quantity") == 0).then(0.0).otherwise(pl.col("AccruedInterest") / pl.col("Quantity"))
            ).alias(output_column)
        )


class FixedRateBondValuationMethod(ValuationMethod):
    """
    Optimized version that builds all cashflows in one table,
    performs a single discount rate lookup, then aggregates.
    """

    @classmethod
    def _correction_method(cls) -> ScalingMethod:
        return MultiplicativeScaling()

    @classmethod
    def calculated_dirty_price(
        cls,
        data: pl.DataFrame,
        projection_date: datetime.date,
        zero_rates: pd.DataFrame,
        output_column: str,
    ) -> pl.DataFrame:
        if data.height == 0:
            return data.with_columns(pl.lit(None, dtype=pl.Float64).alias(output_column))

        # Add row index to preserve order
        data = data.with_row_index(name="_bond_idx")

        # Calculate number of coupons for each bond
        data = data.with_columns(
            NumberOfCoupons=FrequencyRegistry.number_due(pl.col("NextCouponDate"), pl.col("MaturityDate"))
        )

        max_coupons_value = data["NumberOfCoupons"].max()
        if max_coupons_value is None:
            max_coupons = 0
        else:
            max_coupons = int(float(max_coupons_value))  # type: ignore[arg-type]

        # Build all cashflows at once
        cashflow_parts = []

        for i in range(max_coupons + 1):
            # Calculate coupon date for this period
            coupon_date = FrequencyRegistry.step_coupon_date(projection_date, pl.col("MaturityDate"), i)
            years_to_coupon = (coupon_date - pl.lit(projection_date)).dt.total_days() / 365.25

            # Create cashflow records for bonds that have this coupon
            part = data.filter(pl.col("NumberOfCoupons") >= i).with_columns(
                _years=years_to_coupon, _payment=(pl.col("InterestRate") * FrequencyRegistry.portion_year())
            )
            cashflow_parts.append(part)

        # Combine all cashflows into one table
        all_cashflows = pl.concat(cashflow_parts)

        # Get discount factors for ALL cashflows in ONE operation
        discount_factors = get_discount_rates(all_cashflows, zero_rates, pl.col("_years"))
        all_cashflows = all_cashflows.with_columns(pl.Series("_discount_factor", discount_factors))

        # Calculate present value for each cashflow
        all_cashflows = all_cashflows.with_columns(_pv=(pl.col("_payment") * pl.col("_discount_factor")))

        # Sum present values by bond
        result = all_cashflows.group_by("_bond_idx").agg(pl.col("_pv").sum().alias(output_column))

        # Join back to original data
        data = data.join(result, on="_bond_idx", how="left")

        # Add principal repayment at maturity (1.0 * discount factor at maturity)
        years_to_maturity = (pl.col("MaturityDate") - pl.lit(projection_date)).dt.total_days() / 365.25
        maturity_df = get_discount_rates(data, zero_rates, years_to_maturity)
        data = data.with_columns((pl.col(output_column) + maturity_df).alias(output_column))

        return data.drop("_bond_idx", "NumberOfCoupons")


class FloatingRateBondValuationMethod(ValuationMethod):
    @classmethod
    def _correction_method(cls) -> ScalingMethod:
        return MultiplicativeScaling()

    @classmethod
    def calculated_dirty_price(
        cls,
        data: pl.DataFrame,
        projection_date: datetime.date,
        zero_rates: pd.DataFrame,
        output_column: str,
    ) -> pl.DataFrame:
        # include principal (par) for floating rate note
        return _price_spread_instrument(
            data=data,
            projection_date=projection_date,
            zero_rates=zero_rates,
            output_column=output_column,
            rate_expr=pl.col("Spread"),  # spread over float
            include_par=True,
        )


class SwapValuationMethod(ValuationMethod):
    """
    Receive floating, pay fixed (stored in Spread column as typically negative).
    Same mechanics as floating rate bond valuation but WITHOUT principal redemption.
    """

    @classmethod
    def _correction_method(cls) -> ScalingMethod:
        return AdditiveScaling()

    @classmethod
    def calculated_dirty_price(
        cls,
        data: pl.DataFrame,
        projection_date: datetime.date,
        zero_rates: pd.DataFrame,
        output_column: str,
    ) -> pl.DataFrame:
        # no principal payment => include_par = False
        return _price_spread_instrument(
            data=data,
            projection_date=projection_date,
            zero_rates=zero_rates,
            output_column=output_column,
            rate_expr=pl.col("Spread"),  # fixed leg rate (usually negative)
            include_par=False,
        )


def get_discount_rates(
    loans: pl.DataFrame,
    zero_rates: pd.DataFrame,
    time_expr: pl.Expr,
) -> pl.Series:
    """
    Add linear-interpolated zero rate at each loan's maturity and compute discount factor.
    Returns a Series of discount factors.
    """
    lf = loans.lazy() if isinstance(loans, pl.DataFrame) else loans

    # Convert and prepare zero curve once
    zero_pl = (
        pl.from_pandas(zero_rates)
        .select(
            pl.col("Name").alias("zc_name").cast(pl.Utf8),
            pl.col("MaturityYears").alias("zc_t").cast(pl.Float64),
            pl.col("Rate").alias("zc_r").cast(pl.Float64),
        )
        .sort(["zc_name", "zc_t"])
        .lazy()
    )

    # Single pass: compute YearsToMat, add index, and sort
    base = (
        lf.with_columns(YearsToMat=time_expr).with_row_index(name="_idx").sort("YearsToMat")  # Sort once upfront
    )

    # Get lower bound
    with_lower = base.join_asof(zero_pl, left_on="YearsToMat", right_on="zc_t", strategy="backward").rename(
        {"zc_t": "t0", "zc_r": "r0"}
    )

    # Get upper bound - need to sort again after first join
    with_bounds = with_lower.sort("YearsToMat").join_asof(  # Re-sort for second join_asof
        zero_pl.select(pl.col("zc_name"), pl.col("zc_t").alias("t1"), pl.col("zc_r").alias("r1")),
        left_on="YearsToMat",
        right_on="t1",
        strategy="forward",
    )

    # Compute interpolated rate and discount factor in one go
    result = (
        with_bounds.with_columns(
            [
                # Simplified interpolation
                pl.when(pl.col("t0").is_null())
                .then(pl.col("r1"))  # Below min tenor
                .when(pl.col("t1").is_null())
                .then(pl.col("r0"))  # Above max tenor
                .when(pl.col("t1") == pl.col("t0"))
                .then(pl.col("r0"))  # Exact match
                .otherwise(
                    # Linear interpolation
                    pl.col("r0")
                    + (pl.col("r1") - pl.col("r0"))
                    * (pl.col("YearsToMat") - pl.col("t0"))
                    / (pl.col("t1") - pl.col("t0"))
                )
                .alias("rate")
            ]
        )
        .with_columns(
            [
                # Compute discount factor
                (-pl.col("rate") * pl.col("YearsToMat")).exp().alias("dfs")
            ]
        )
        .sort("_idx")
    )

    # Collect once at the end and return only the discount factor series
    return result.select("dfs").collect().to_series()


# Shared helper for floating-like instruments (floating notes, swaps)
def _price_spread_instrument(
    data: pl.DataFrame,
    projection_date: datetime.date,
    zero_rates: pd.DataFrame,
    output_column: str,
    rate_expr: pl.Expr,
    include_par: bool,
) -> pl.DataFrame:
    """
    Optimized present value builder using vectorized approach:
      start value = (par if include_par) + accrued
      add sum_i rate_expr * Δ_i * DF_i for remaining periods
    Builds all cashflows at once, then does single discount rate lookup.
    """
    if data.height == 0:
        return data.with_columns(pl.lit(None, dtype=pl.Float64).alias(output_column))

    # Add row index and calculate number of coupons
    data = data.with_row_index(name="_inst_idx").with_columns(
        NumberOfCoupons=FrequencyRegistry.number_due(pl.col("NextCouponDate"), pl.col("MaturityDate"))
    )

    max_coupons_value = data["NumberOfCoupons"].max()
    if max_coupons_value is None:
        max_coupons = 0
    else:
        max_coupons = int(float(max_coupons_value))  # type: ignore[arg-type]

    # Build all cashflows at once
    cashflow_parts = []

    for i in range(max_coupons + 1):
        # Calculate coupon date for this period
        coupon_date = FrequencyRegistry.step_coupon_date(projection_date, pl.col("MaturityDate"), i)
        years_to_coupon = (coupon_date - pl.lit(projection_date)).dt.total_days() / 365.25

        # Calculate previous coupon date to get delta (period length)
        if i == 0:
            prev_coupon_date = pl.col("PreviousCouponDate")
        else:
            prev_coupon_date = FrequencyRegistry.step_coupon_date(projection_date, pl.col("MaturityDate"), i - 1)

        delta = (coupon_date - prev_coupon_date).dt.total_days() / 365.25

        # Create cashflow records for instruments that have this coupon
        part = data.filter(pl.col("NumberOfCoupons") >= i).with_columns(
            _years=years_to_coupon, _payment=rate_expr * delta
        )
        cashflow_parts.append(part)

    # Combine all cashflows into one table
    all_cashflows = pl.concat(cashflow_parts)

    # Get discount factors for ALL cashflows in ONE operation
    discount_factors = get_discount_rates(all_cashflows, zero_rates, pl.col("_years"))
    all_cashflows = all_cashflows.with_columns(pl.Series("_discount_factor", discount_factors))

    # Calculate present value for each cashflow
    all_cashflows = all_cashflows.with_columns(_pv=pl.col("_payment") * pl.col("_discount_factor"))

    # Sum present values by instrument
    result = all_cashflows.group_by("_inst_idx").agg(pl.col("_pv").sum().alias(output_column))

    # Join back to original data
    data = data.join(result, on="_inst_idx", how="left")

    # Add starting value (principal + accrued interest)
    accrued = pl.when(pl.col("Quantity") == 0).then(0.0).otherwise(pl.col("AccruedInterest") / pl.col("Quantity"))
    start_val = accrued
    if include_par:
        # Add principal at maturity for floating rate bonds
        years_to_maturity = (pl.col("MaturityDate") - pl.lit(projection_date)).dt.total_days() / 365.25
        maturity_df = get_discount_rates(data, zero_rates, years_to_maturity)
        start_val = start_val + maturity_df

    data = data.with_columns((pl.col(output_column) + start_val).alias(output_column))

    return data.drop("_inst_idx", "NumberOfCoupons")


class ValuationMethodRegistry(BaseRegistry[ValuationMethod]):
    @classmethod
    def corrected_dirty_price(
        cls,
        data: pl.DataFrame,
        projection_date: datetime.date,
        zero_rates: pd.DataFrame,
        output_column: str,
    ) -> pl.DataFrame:
        results = []
        for (valuation_method,), valuation_method_data in data.partition_by("ValuationMethod", as_dict=True).items():
            method = cls.get(str(valuation_method))
            results.append(
                method.corrected_dirty_price(valuation_method_data, projection_date, zero_rates, output_column)
            )

        return pl.concat(results)


ValuationMethodRegistry.register("none", NoValuationMethod())
ValuationMethodRegistry.register("amortizedcost", AmortizedCostValuationMethod())
ValuationMethodRegistry.register("fixedratebond", FixedRateBondValuationMethod())
ValuationMethodRegistry.register("floatingratebond", FloatingRateBondValuationMethod())
ValuationMethodRegistry.register("swap", SwapValuationMethod())
