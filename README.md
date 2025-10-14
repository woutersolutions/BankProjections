# BankProjections

A comprehensive balance sheet projection tool for banks, designed for capital management, liquidity management, stress
testing, and planning.

## Overview

BankProjections enables financial institutions to:
- Project financials statements into the future
- Apply scenario-based projections with customizable time intervals
- Calculate amortization schedules for balance sheet items
- Compute regulatory metrics (capital ratios, ROC, LCR, NSFR)
- Output future balance sheet, regulatory metrics, income statement, and cashflow statement

## Features

- **Multi-scenario Support**: Run multiple projection scenarios
- **Flexible Input**: High-level or detailed balance sheet table and flexible scenario input templates.
- **Flexible Calculations**: Orchestrate calculation details using classification columns on the balance sheet and
  registries.
- **Runoff**: Calculate repayments, pre-payments, and coupon payements
- **Valuation**: Market value calculations based on scenario curves.
- **Regulatory Compliance**: Calculate Basel III capital ratios and liquidity metrics
- **Synthetic Data Generation**: Test and demonstrate capabilities without real bank data

Built on Polars, it’s extremely fast and capable of handling portfolios with millions of loans. It is highly flexible to
adapt to any bank setup.

## Installation

### Development Setup

The code is developed in Python 3.13. See configuration in pyproject.toml for more details.

```bash
# Clone the repository
git clone https://github.com/woutersolutions/BankProjections.git
cd BankProjections

# Install in development mode with all dependencies
pip install -e ".[dev]"
```

## Usage

For an example, see the `src/examples/` directory. It contains a script `main.py` that demonstrates how to use the
library. It is based on synthetic data confidered in `example_bs.csv` and `example_params.yaml`, and uses an example
scenario input file `example_scenario.xlsx`. The script generate a synthethic balance sheet, perform the projections and
metric calculations, and output the results to Excel.

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

The project maintains high code quality standards:

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

This tool is work in progress and not production-ready yet. It needs to be tailored to the specific needs of each bank
and should be used with caution. Always validate results against known benchmarks.