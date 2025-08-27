# Hierarchical Time Series Forecasting

This project provides a comprehensive pipeline for hierarchical time series forecasting. It leverages the `hierarchicalforecast` package to build, evaluate, and compare various forecasting models, including a novel graph-based neural network. The primary goal is to provide a robust framework for handling both temporal and geographical aggregations in time series data, with a focus on probabilistic predictions and evaluation.

## Features

- **End-to-End Pipeline**: From data loading and preprocessing to model training, prediction, and evaluation.
- **Hierarchical Forecasting**: Supports both temporal and geographical (or other categorical) hierarchies.
- **Probabilistic Predictions**: Generates probabilistic forecasts to capture uncertainty.
- **Graph-Based Neural Network**: Implements a custom graph-based neural network for comparison with baseline models.
- **Modular and Scalable**: The codebase is structured to be easily extensible and scalable.

## Project Structure

```
hierarchical_ts/
├── data/                 # Raw and processed data
├── models/               # Trained models
├── scripts/              # Scripts to run the pipeline
│   └── run_pipeline.py
├── src/                  # Source code
│   └── hierarchical_ts/
│       ├── __init__.py
│       ├── data.py       # Data loading and preprocessing
│       ├── models.py     # Forecasting models
│       ├── train.py      # Model training
│       ├── predict.py    # Prediction pipeline
│       ├── evaluate.py   # Evaluation metrics
│       └── utils.py      # Utility functions
├── tests/                # Unit tests
├── .gitignore
├── pyproject.toml        # Project metadata and dependencies
├── setup.cfg             # Package configuration
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.8+
- Poetry (for dependency management)

### Installation

1. Clone the repository:
   ```bash
   git clone https://your-repo-url/hierarchical_ts.git
   cd hierarchical_ts
   ```

2. Install the dependencies:
   ```bash
   pip install .
   ```

### Usage

To run the full pipeline, execute the `run_pipeline.py` script:

```bash
python scripts/run_pipeline.py
```

This will:
1. Load and preprocess the data.
2. Train the forecasting models.
3. Generate predictions.
4. Evaluate the models and save the results.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
