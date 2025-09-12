import datetime

from bank_projections.projections.projection import Projection
from bank_projections.projections.runoff import Runoff
from bank_projections.projections.time import TimeHorizon
from examples.synthetic_data import create_synthetic_balance_sheet

if __name__ == "__main__":
    start_date = datetime.date(2024, 12, 31)
    start_bs = create_synthetic_balance_sheet(start_date)
    rules = [Runoff()]
    horizon = TimeHorizon.from_numbers(
        start_date=start_date,
        number_of_days=7,
        number_of_weeks=4,
        number_of_months=12,
        number_of_years=5,
        end_of_month=True,
    )

    projection = Projection(start_bs, rules, horizon)

    projection.run()
