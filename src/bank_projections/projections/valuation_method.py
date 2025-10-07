import datetime
from abc import ABC, abstractmethod

import pandas as pd
import polars as pl

from bank_projections.projections.base_registry import BaseRegistry
from bank_projections.projections.frequency import FrequencyRegistry


class ValuationMethod(ABC):
    @classmethod
    @abstractmethod
    def dirty_price(
        cls,
        data: pl.DataFrame,
        projection_date: datetime.date,
        zero_rates: pd.DataFrame,
        output_column: str,
    ) -> pl.DataFrame:
        pass


class NoValuationMethod(ValuationMethod):
    @classmethod
    def dirty_price(
        cls,
        data: pl.DataFrame,
        projection_date: datetime.date,
        zero_rates: pd.DataFrame,
        output_column: str,
    ) -> pl.DataFrame:
        return data.with_columns(pl.lit(None, dtype=pl.Float64).alias(output_column))


class AmortizedCostValuationMethod(ValuationMethod):
    @classmethod
    def dirty_price(
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
    def dirty_price(
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

        for i in range(data["NumberOfCoupons"].max() + 1):
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


def get_discount_rates(
    loans: pl.DataFrame,
    zero_rates: pd.DataFrame,
    time_expr: pl.Expr,
) -> pl.Series:
    """
    Add linear-interpolated zero rate at each loan's maturity and (optionally) a discount factor.
    Returns a LazyFrame with added columns `out_rate_col` (and `out_df_col` if requested).
    """
    lf = loans.lazy() if isinstance(loans, pl.DataFrame) else loans

    # TODO: Performance improvements
    # TODO: Move this to market data class

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

    # Years to maturity
    base = lf.with_columns(YearsToMat=time_expr).with_row_index(name="_idx").sort("YearsToMat")

    # Lower bracket (t0,r0): last curve point <= t
    lower = base.join_asof(zero_pl, left_on="YearsToMat", right_on="zc_t", strategy="backward").select(
        pl.all(),
        pl.col("zc_t").alias("t0"),
        pl.col("zc_r").alias("r0"),
    )

    # Upper bracket (t1,r1): first curve point >= t
    upper = base.join_asof(zero_pl, left_on="YearsToMat", right_on="zc_t", strategy="forward").select(
        pl.all(),
        pl.col("zc_t").alias("t1"),
        pl.col("zc_r").alias("r1"),
    )

    # Combine & interpolate
    rates = (
        lower.join(
            upper.select("_idx", "YearsToMat", "t1", "r1"),
            on=["_idx", "YearsToMat"],
            how="inner",
        )
        .sort("_idx")
        .with_columns(
            # linear interpolation with sensible edge handling
            pl.when(pl.col("t0").is_not_null() & pl.col("t1").is_not_null() & (pl.col("t1") == pl.col("t0")))
            .then(pl.coalesce([pl.col("r0"), pl.col("r1")]))  # both rates equal
            .when(pl.col("t0").is_not_null() & pl.col("t1").is_not_null() & (pl.col("t1") != pl.col("t0")))
            .then(
                pl.col("r0")
                + (pl.col("r1") - pl.col("r0")) * (pl.col("YearsToMat") - pl.col("t0")) / (pl.col("t1") - pl.col("t0"))
            )
            .when(pl.col("t0").is_not_null() & pl.col("t1").is_null())  # above max tenor
            .then(pl.col("r0"))
            .when(pl.col("t1").is_not_null() & pl.col("t0").is_null())  # below min tenor
            .then(pl.col("r1"))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("rate"),
        )
        .select(
            pl.col("_idx"),
            pl.col("YearsToMat"),
            pl.col("rate"),
            (-pl.col("rate") * pl.col("YearsToMat")).exp().alias("dfs"),
        )
    ).collect()

    return base.collect().join(rates, on="_idx", how="left").sort("_idx").select(pl.col("dfs")).to_series()


class ValuationMethodRegistry(BaseRegistry[ValuationMethod], ValuationMethod):
    @classmethod
    def dirty_price(
        cls,
        data: pl.DataFrame,
        projection_date: datetime.date,
        zero_rates: pd.DataFrame,
        output_column: str,
    ) -> pl.DataFrame:
        results = []
        for (valuation_method,), valuation_method_data in data.partition_by("ValuationMethod", as_dict=True).items():
            method = cls.get(valuation_method)
            results.append(method.dirty_price(valuation_method_data, projection_date, zero_rates, output_column))

        return pl.concat(results)


ValuationMethodRegistry.register("none", NoValuationMethod())
ValuationMethodRegistry.register("amortizedcost", AmortizedCostValuationMethod())
ValuationMethodRegistry.register("fixedratebond", FixedRateBondValuationMethod())
