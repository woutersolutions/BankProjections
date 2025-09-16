# BankProjections

A comprehensive balance sheet projection tool for banks, designed for capital management, liquidity management, and stress testing.

## Overview

BankProjections enables financial institutions to:
- Construct financial statements from various input formats (YAML, CSV, Excel)
- Project financials statements into the future
- Apply scenario-based projections with customizable time intervals
- Calculate amortization schedules for balance sheet items
- Compute regulatory metrics (capital ratios, ROC, LCR, NSFR)
- Generate comprehensive output reports

## Features

- **Multi-scenario Support**: Run multiple projection scenarios simultaneously
- **Flexible Input**: Accept data in YAML, CSV, or Excel formats
- **Regulatory Compliance**: Calculate Basel III capital ratios and liquidity metrics
- **Synthetic Data Generation**: Test and demonstrate capabilities without real bank data

## Installation

### Development Setup

```bash
# Clone the repository
git clone https://github.com/woutersolutions/BankProjections.git
cd BankProjections

# Install in development mode with all dependencies
pip install -e ".[dev]"
```

## Usage

Here's a simple example using the current API:

```python
import datetime
from bank_projections.projections.projection import Projection
from bank_projections.projections.runoff import Runoff
from bank_projections.projections.time import TimeHorizon
from examples.synthetic_data import create_synthetic_balance_sheet

# Create a synthetic balance sheet
start_date = datetime.date(2024, 12, 31)
balance_sheet = create_synthetic_balance_sheet(start_date)

# Define projection rules
rules = [Runoff()]

# Create time horizon
horizon = TimeHorizon.from_numbers(
    start_date=start_date,
    number_of_months=12,
    number_of_years=2,
    end_of_month=True,
)

# Run projection
projection = Projection(rules, horizon)
results = projection.run(balance_sheet)
```

For more complete examples, see the `src/examples/` directory.

## Project Structure

```
src/bank_projections/
├── data_preparation/   # Input parsing and validation
├── amortization/       # Amortization calculations
├── projections/        # Scenario-based forecasting
├── metrics/           # Financial metrics and ratios
└── output/            # Result formatting and export

tests/                 # Unit and integration tests
docs/                  # Documentation
examples/              # Sample data and scenarios
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/unit/test_projections.py
```

### Code Quality

```bash
# Format code
ruff format .

# Check linting
ruff check .

# Type checking
mypy src/
```

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please ensure all tests pass and code meets quality standards before submitting pull requests.

## Disclaimer

This tool uses synthetic data for testing and demonstration purposes only.