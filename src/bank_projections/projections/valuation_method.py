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
        data = data.with_columns(
            **{
                "YearsToMaturity": (pl.col("MaturityDate") - pl.lit(projection_date)).dt.total_days() / 365.25,
                "NumberOfCoupons": FrequencyRegistry.number_due(pl.col("NextCouponDate"), pl.col("MaturityDate")),
            }
        )

        data = data.with_columns(get_discount_rates(data, zero_rates, pl.col("YearsToMaturity")).alias(output_column))

        max_coupons_value = data["NumberOfCoupons"].max()
        if max_coupons_value is None:
            max_coupons = 0
        else:
            # Cast to float first, handling polars scalar types
            max_coupons = int(float(max_coupons_value))  # type: ignore[arg-type]
        for i in range(max_coupons + 1):
            coupon_date = FrequencyRegistry.step_coupon_date(projection_date, pl.col("MaturityDate"), i)
            years_to_coupon = (coupon_date - pl.lit(projection_date)).dt.total_days() / 365.25
            discount_rates_i = get_discount_rates(data, zero_rates, years_to_coupon)
            payment_i = (
                pl.when(pl.col("NumberOfCoupons") >= i)
                .then(pl.col("InterestRate") * FrequencyRegistry.portion_year())
                .otherwise(0.0)
            )

            assert (
                data.with_columns(discount_rates_i.alias("test"))
                .select(pl.col("test").is_not_null().all())
                .to_series()[0]
            ), "Discounted price calculation error"

            data.with_columns(
                (pl.col(output_column) + payment_i * (-discount_rates_i * years_to_coupon).exp()).alias(output_column)
            )

        assert (
            data.with_columns(data[output_column].alias("test"))
            .select(pl.col("test").is_not_null().all())
            .to_series()[0]
        ), "Discounted price calculation error"

        return data.drop("YearsToMaturity", "NumberOfCoupons")


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
    Generic present value builder:
      start value = (par if include_par) + accrued
      add sum_i rate_expr * Δ_i * DF_i for remaining periods
    Assumes:
      - 'Spread' (or supplied rate_expr) per period
      - Frequency data via FrequencyRegistry
      - Accrued interest provided (quantity-scaled externally)
    """
    if data.height == 0:
        return data.with_columns(pl.lit(None, dtype=pl.Float64).alias(output_column))

    df = data.with_columns(
        NumberOfCoupons=FrequencyRegistry.number_due(pl.col("NextCouponDate"), pl.col("MaturityDate")),
        _acc=pl.when(pl.col("Quantity") == 0).then(0.0).otherwise(pl.col("AccruedInterest") / pl.col("Quantity")),
    )

    start_val = pl.lit(0.0)
    if include_par:
        start_val = start_val + pl.lit(1.0)
    start_val = start_val + pl.col("_acc")

    out = df.with_columns(start_val.alias(output_column))

    max_cpn_value = out["NumberOfCoupons"].max() if out.height > 0 else None
    max_cpn = 0 if max_cpn_value is None else int(float(max_cpn_value))  # type: ignore[arg-type]
    for i in range(max_cpn + 1):
        cpn_date_i = FrequencyRegistry.step_coupon_date(projection_date, pl.col("MaturityDate"), i)
        dt_years_i = (cpn_date_i - pl.lit(projection_date)).dt.total_days() / 365.25
        dfs_i = get_discount_rates(out, zero_rates, dt_years_i)

        if i == 0:
            prev_date_i = pl.col("PreviousCouponDate")
        else:
            prev_date_i = FrequencyRegistry.step_coupon_date(projection_date, pl.col("MaturityDate"), i - 1)
        delta_i = (cpn_date_i - prev_date_i).dt.total_days() / 365.25

        contrib_i = pl.when(pl.col("NumberOfCoupons") >= i).then(rate_expr * delta_i * dfs_i).otherwise(0.0)
        out = out.with_columns((pl.col(output_column) + contrib_i).alias(output_column))

    return out.drop("_acc", "NumberOfCoupons")


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
