from typing import Any

import polars as pl

from bank_projections.config import Config
from bank_projections.projections.frequency import FrequencyRegistry
from bank_projections.utils.date import add_months, is_end_of_month
from bank_projections.utils.time import TimeHorizon


def calculate_profitability(
    metric_list: list[dict[str, float]], pnl_list: list[pl.DataFrame], horizon: TimeHorizon
) -> list[dict[str, Any]]:
    result_list: list[dict[str, Any]] = []
    current_date = horizon.dates[len(metric_list) - 1]
    if not is_end_of_month(current_date):
        return []

    for outlook in Config.PROFITABILITY_OUTLOOKS:
        # Config.PROFITABILITY_OUTLOOKS only contains monthly-based frequencies (Monthly, Quarterly, Annual)
        number_of_months = FrequencyRegistry.get(outlook).number_of_months  # type: ignore[attr-defined]
        horizon_start_date = add_months(current_date, -number_of_months, make_end_of_month=True)
        find_index = [i for i, date in enumerate(horizon.dates) if date == horizon_start_date]
        if len(find_index) == 0:
            continue
        start_index = find_index[0]

        # Calculate weighted average metrics
        wa_metrics = {}
        total_days = (current_date - horizon_start_date).days
        for metric in metric_list[0]:
            wa_metrics[metric] = float(
                sum(
                    metric_list[i][metric] * (horizon.dates[i + 1] - horizon.dates[i]).days / total_days
                    for i in range(start_index, len(metric_list) - 1)
                )
            )

        pnls = pl.concat(pnl_list[(start_index + 1) :])
        result = calculate_profitability_single_horizon(wa_metrics, pnls, number_of_months)
        result_list.append({"outlook": outlook, **result})
    return result_list


def calculate_profitability_single_horizon(
    wa_metrics: dict[str, float], pnls: pl.DataFrame, number_of_months: int
) -> dict[str, Any]:
    result = {
        "Total Assets": wa_metrics["Total Assets"],
        "Total Equity": wa_metrics["Total Equity"],
        "Net Income": pnls["Amount"].sum(),
        "Net Interest Income": pnls.filter(pl.col("rule").is_in(["Accrual", "Coupons"]))
        .select(pl.col("Amount").sum())
        .item(),
    }

    result["Return on Assets"] = annualize(result["Net Income"] / wa_metrics["Total Assets"], number_of_months)
    result["Return on Equity"] = annualize(result["Net Income"] / wa_metrics["Total Equity"], number_of_months)
    result["Net Interest Margin"] = annualize(
        result["Net Interest Income"] / wa_metrics["Total Assets"], number_of_months
    )

    return result


def annualize(value: float, number_of_months: int) -> float:
    return float((1 + value) ** (12 / number_of_months) - 1)
