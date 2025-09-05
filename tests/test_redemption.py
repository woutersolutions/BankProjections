import datetime

import polars as pl

from bank_projections.projections.frequency import Annual, FrequencyRegistry
from bank_projections.projections.redemption import (
    AnnuityRedemption,
    BulletRedemption,
    LinearRedemption,
    PerpetualRedemption,
    RedemptionRegistry,
)


class TestBulletRedemption:
    def test_bullet_before_maturity(self):
        maturity = pl.lit(datetime.date(2025, 12, 31))
        rate = pl.lit(0.05)
        coupon_date = pl.lit(datetime.date(2024, 12, 31))
        projection = datetime.date(2025, 6, 30)

        factor = BulletRedemption.redemption_factor(maturity, rate, coupon_date, projection)
        df = pl.DataFrame({"dummy": [1]}).with_columns(factor.alias("factor"))

        assert df["factor"][0] == 0.0

    def test_bullet_at_maturity(self):
        maturity = pl.lit(datetime.date(2025, 6, 30))
        rate = pl.lit(0.05)
        coupon_date = pl.lit(datetime.date(2024, 6, 30))
        projection = datetime.date(2025, 6, 30)

        factor = BulletRedemption.redemption_factor(maturity, rate, coupon_date, projection)
        df = pl.DataFrame({"dummy": [1]}).with_columns(factor.alias("factor"))

        assert df["factor"][0] == 1.0

    def test_bullet_after_maturity(self):
        maturity = pl.lit(datetime.date(2025, 6, 30))
        rate = pl.lit(0.05)
        coupon_date = pl.lit(datetime.date(2024, 6, 30))
        projection = datetime.date(2025, 12, 31)

        factor = BulletRedemption.redemption_factor(maturity, rate, coupon_date, projection)
        df = pl.DataFrame({"dummy": [1]}).with_columns(factor.alias("factor"))

        assert df["factor"][0] == 1.0


class TestPerpetualRedemption:
    def test_perpetual_no_redemption(self):
        maturity = pl.lit(datetime.date(2025, 12, 31))
        rate = pl.lit(0.05)
        coupon_date = pl.lit(datetime.date(2024, 12, 31))
        projection = datetime.date(2025, 6, 30)

        factor = PerpetualRedemption.redemption_factor(maturity, rate, coupon_date, projection)
        df = pl.DataFrame({"dummy": [1]}).with_columns(factor.alias("factor"))

        assert df["factor"][0] == 0.0


class TestLinearRedemption:
    def test_linear_constant_principal_payment(self):
        maturity = pl.lit(datetime.date(2025, 12, 31))
        rate = pl.lit(0.05)
        coupon_date = pl.lit(datetime.date(2023, 12, 31))
        projection = datetime.date(2023, 12, 31)  # Start of loan

        # Create test dataframe with required columns
        df = pl.DataFrame({"CouponFrequency": ["annual"], "dummy": [1]}).with_columns(
            [
                LinearRedemption.redemption_factor(maturity, rate, coupon_date, projection).alias("factor"),
                FrequencyRegistry.number_due(coupon_date, pl.lit(projection)).alias("payments_left"),
            ]
        )

        payments_left = df["payments_left"][0]
        expected_factor = 1.0 / payments_left if payments_left > 0 else 0.0

        assert abs(df["factor"][0] - expected_factor) < 1e-10


class TestAnnuityRedemption:
    def test_annuity_payment_formula(self):
        maturity = pl.lit(datetime.date(2025, 12, 31))
        rate = pl.lit(0.05)
        coupon_date = pl.lit(datetime.date(2023, 12, 31))
        projection = datetime.date(2023, 12, 31)  # Start of loan

        # Create test dataframe with required columns
        df = pl.DataFrame({"CouponFrequency": ["annual"], "dummy": [1]}).with_columns(
            [
                AnnuityRedemption.redemption_factor(maturity, rate, coupon_date, projection).alias("factor"),
                FrequencyRegistry.number_due(coupon_date, pl.lit(projection)).alias("payments_left"),
                FrequencyRegistry.portion_year().alias("period_fraction"),
            ]
        )

        payments_left = df["payments_left"][0]
        period_rate = 0.05 * df["period_fraction"][0]
        expected_factor = period_rate / (1 - (1 + period_rate) ** (-payments_left)) if payments_left > 0 else 0.0

        assert abs(df["factor"][0] - expected_factor) < 1e-10


class TestRedemptionRegistry:
    def test_registry_routing(self):
        # Set up frequency registry for testing
        FrequencyRegistry.register("annual", Annual)

        maturity = pl.lit(datetime.date(2025, 12, 31))
        rate = pl.lit(0.05)
        coupon_date = pl.lit(datetime.date(2024, 12, 31))
        projection = datetime.date(2025, 6, 30)

        # Test bullet routing
        df_bullet = pl.DataFrame({"RedemptionType": ["bullet"], "CouponFrequency": ["annual"]}).with_columns(
            RedemptionRegistry.redemption_factor(maturity, rate, coupon_date, projection).alias("factor")
        )

        # Test perpetual routing
        df_perpetual = pl.DataFrame({"RedemptionType": ["perpetual"], "CouponFrequency": ["annual"]}).with_columns(
            RedemptionRegistry.redemption_factor(maturity, rate, coupon_date, projection).alias("factor")
        )

        assert df_bullet["factor"][0] == 0.0  # Before maturity
        assert df_perpetual["factor"][0] == 0.0  # Never redeems
