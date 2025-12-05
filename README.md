# BankProjections

A powerful, open-source balance sheet projection engine for banks and financial institutions. Built for speed,
flexibility, and transparency in capital management, liquidity forecasting, and stress testing.

## Why BankProjections?

Traditional banking projection are often in various complex Excel sheets. BankProjections offers:

- **🚀 Blazing Fast**: Built on Polars, handles portfolios with millions of instruments efficiently
- **🔧 Highly Flexible**: Adapt calculations to your bank's specific needs using configurable registries
- **📊 Comprehensive**: Projects balance sheets, P&L, cash flows, and regulatory metrics
- **🎯 Scenario-Based**: Run multiple concurrent scenarios with customizable assumptions
- **🔓 Open Source**: Full transparency with Apache License 2.0
- **🧪 Test-Driven**: Synthetic data generation for safe testing and demonstration

## What It Does

BankProjections enables financial institutions to:

- Project financial statements into the future under various scenarios
- Calculate runoff patterns, amortization, and repayment schedules
- Compute regulatory metrics (Basel IV capital ratios, LCR, NSFR, ROC)
- Perform market valuations based on scenario curves
- Generate income statements and cash flow projections
- Stress test portfolios under adverse conditions

## Key Features

### Core Capabilities

- **Multi-scenario Support**: Run multiple projection scenarios concurrently with independent assumptions
- **Runoff Modeling**: Calculate repayments, prepayments, and coupon payments with custom amortization rules
- **Market Valuation**: Dynamic market value calculations based on scenario yield curves
- **Regulatory Compliance**: Basel IV capital ratios, LCR, NSFR, and other regulatory metrics
- **Synthetic Data Generation**: Built-in tools to generate realistic test data without exposing real bank information

> **⚠️ Work in Progress**: This project is under active development and not yet production-ready. Contributions and
> feedback are welcome!

### Flexible Architecture via Registries

BankProjections uses a **registry-based architecture** that allows you to customize calculations without modifying core
code:

A few examples:

1. **BalanceSheetItemRegistry**: Define important balance sheet items that should be used in calculations
2. **BalanceSheetMetrics**: Configure which metrics are calculated and how
    - Standard metrics: nominal, book value, interest rates, impairments
    - Add custom metrics specific to your institution
    - Control metric calculation order and dependencies
3. **AccountingMethodRegistry**: Define how book value is determined and income is accounted:
    - Amortized cost
    - Fair value through P&L
    - Fair value through OCI
4. **MetricRegistry**: Defines the output metrics, for example regulatory capital and liquidity ratios
5. **ValuationRegistry**: Defines how items must be reevaluated
6. **RedemptionRegistry**: Defines how items redeem (principal payments)

Most registries correspond to a column in the input balance sheet containing the registry keys. This way the relevant
calculations can be orchestrated for each balance item.

This registry approach means you can adapt BankProjections to your bank's specific needs by **registering custom
handlers** rather than forking and modifying the core codebase. Simply extend the registries with your
institution-specific logic while maintaining compatibility with updates.

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

Make sure to add the `src` directory to your `PYTHONPATH`.

## Quick Start

### Basic Usage

See the `src/examples/` directory for a complete working example:

```bash
# Navigate to examples
cd src/examples

# Run the example projection
python main.py
```

The example demonstrates:

- Loading a synthetic balance sheet (`example_bs.csv`)
- Applying scenario parameters `example_scenario.xlsx`
- Running a projection with this scenario `main.py`
- Computing regulatory metrics
- Exporting results to Excel

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

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contributing, contact & Support

If you have feedback, questions or offer/need help:

- **Issues**: For bug reports or feature requests,
  please [open an issue](https://github.com/woutersolutions/BankProjections/issues)
- **LinkedIn**: [Wouter van Krieken](https://www.linkedin.com/in/wouter-van-krieken/)

## Project Status & Roadmap

**Current Status**: Alpha - Core functionality is implemented but the API may change

### Completed

- ✅ Core projection engine with scenario support
- ✅ Amortization and runoff calculations
- ✅ Excel scenario input templates
- ✅ Basic regulatory metrics (capital ratios, liquidity ratios)
- ✅ Synthetic data generation
- ✅ Registry-based customization system

### In Progress

- 🔨 Enhanced documentation and examples
- 🔨 Additional scenario templates
- 🔨 Performance optimizations for large portfolios
- 🔨 Off-balance modelling and hedging
- 🔨 IFRS9 Stage migrations and impairments changes

### Planned

- 📋 Web-based visualization dashboard
- 📋 Additional regulatory metrics

## Disclaimer

**⚠️ This tool is under active development and not production-ready.**

- Results should be validated against known benchmarks and existing systems
- The tool needs to be tailored to the specific needs of each institution
- No warranty is provided - use at your own risk (see Apache License 2.0)
- Not intended to replace professional financial advice or regulatory compliance systems

Always perform thorough validation and testing before using for any critical decision-making.