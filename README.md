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

# TODO: Demo the usage (the code below is not real) here and refer to the examples folder

```python
from bank_projections import ProjectionEngine

# Initialize the projection engine
engine = ProjectionEngine()

# Load bank data and scenarios
engine.load_data("path/to/balance_sheet.yaml")
engine.load_scenarios("path/to/scenarios.yaml")

# Run projections
results = engine.project(
    start_date="2024-01-01",
    end_date="2025-12-31",
    frequency="quarterly"
)

# Calculate metrics
metrics = engine.calculate_metrics(results)

# Export results
engine.export_results(metrics, "output/projection_results.xlsx")
```

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