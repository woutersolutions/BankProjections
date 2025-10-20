import datetime
import os

from bank_projections.projections.projection import Projection
from bank_projections.projections.runoff import Runoff
from bank_projections.projections.valuation import Valuation
from bank_projections.scenarios.template_registry import TemplateRegistry
from bank_projections.utils.logging import setup_logger_format_with_context
from bank_projections.utils.time import TimeHorizon
from examples import EXAMPLE_FOLDER, OUTPUT_FOLDER
from examples.synthetic_data import create_synthetic_balance_sheet

if __name__ == "__main__":
    start_date = datetime.date(2024, 12, 31)
    setup_logger_format_with_context()

    scenario = TemplateRegistry.load_folder(os.path.join(EXAMPLE_FOLDER, "scenarios"))
    scenario.rules = {"Runoff": Runoff(), "Valuation": Valuation(), **scenario.rules}
    horizon = TimeHorizon.from_numbers(
        start_date=start_date,
        number_of_days=3,
        number_of_months=12,
        number_of_years=3,
        end_of_month=True,
    )

    start_bs = create_synthetic_balance_sheet(start_date, scenario)
    projection = Projection({"base": scenario}, horizon)
    result = projection.run(start_bs)
    result.to_excel(os.path.join(OUTPUT_FOLDER, "main example.xlsx"), open_after=True)
