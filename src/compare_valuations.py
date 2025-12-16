import datetime
import os
import time

import polars as pl

from bank_projections.projections.valuation_method import ValuationMethodRegistry
from bank_projections.scenarios.excel_sheet_format import TemplateTypeRegistry
from bank_projections.scenarios.scenario import Scenario
from bank_projections.utils.time import TimeIncrement
from examples import EXAMPLE_FOLDER
from examples.synthetic_data import generate_synthetic_positions, read_range

methods = ["swap"]

number_of_loans = 10_000

config_table = pl.read_csv(os.path.join(EXAMPLE_FOLDER, "example_bs.csv"))

position_config = config_table.filter(pl.col("sub_item_type") == "Swaps").to_dicts()[0]
position_input = {
    name: read_range(value) if name.endswith("_range") else value for name, value in position_config.items()
}
position_input["number"] = number_of_loans

current_date = datetime.date(2024, 12, 31)
excel_inputs = TemplateTypeRegistry.load_folder(os.path.join(EXAMPLE_FOLDER, "scenarios"))
scenario = Scenario(excel_inputs)
scenario_snapshot = scenario.snapshot_at(TimeIncrement(current_date, current_date))
curves = scenario_snapshot.curves
zero_rates = curves.get_zero_rates()

positions = generate_synthetic_positions(current_date=current_date, curves=curves, **position_input)

for method in methods:
    print(f"Valuation method: {method}")

    start_time = time.perf_counter()

    valuation_method = ValuationMethodRegistry.get(method)
    valuation = valuation_method.calculated_dirty_price(positions._data, current_date, zero_rates, method)

    end_time = time.perf_counter()
    processing_time = end_time - start_time

    print(f"Time taken: {processing_time:.4f} seconds")
    print(f"Total valuation: {valuation[method].sum()}")
