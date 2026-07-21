# F1 Race Position Prediction

Predict Formula One race finishing positions using deep neural networks based on driver performance features.

## Overview

This project uses **deep neural networks** to predict F1 race finishing positions, specifically focusing on **top-10 finishers**. The model is trained on historical F1 data (2020-2024) and uses 9 carefully engineered features to capture driver performance, track history, constructor competitiveness, and current form.

### Key Features

- **Top-10 Focused Model**: Trained exclusively on competitive drivers (positions 1-10) for improved accuracy
- **9 Domain-Specific Features**: Season trends, recent momentum, track expertise, constructor performance, grid position, and track characteristics
- **Time-Aware Cross-Validation**: Prevents temporal data leakage with strict year separation
- **Deep Neural Network Architecture**: 3-layer MLP (128→64→32 neurons) with regularization

### Model Performance

- **Mean Absolute Error**: 1.585 positions on 2025 filtered test data
- **Within 3 positions**: 88.7% accuracy
- **Within 2 positions**: 67.3% accuracy
- **Within 1 position**: 37.9% accuracy
- **Exact match**: 10.2% accuracy

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository** (if applicable) or navigate to the project directory

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Code

#### Step 1: Data Collection (Optional if data already exists)

**Skip this step if the `data/` folder already contains `training_data.csv` and `test_data.csv`.**

If you need to collect data from scratch:

```bash
python collect_data.py
```

This will:
- Download F1 data from the Fast F1 API (first run may take time)
- Collect training data from 2020-2024 seasons
- Collect test data from 2025 season
- Calculate all 9 features for each driver-race combination
- Save data to `data/training_data.csv` and `data/test_data.csv`
- Cache API responses in `cache/` directory for faster subsequent runs

**Note**: Data collection uses caching. After the first run, subsequent runs are much faster as they use cached data.

#### Step 2: Train the Model

Train the top-10 focused neural network model:

```bash
python top10/train.py
```

This will:
- Load training and test data from `data/` folder
- Filter to top-10 finishers only (positions 1-10)
- Filter out DNFs and extreme outliers (finish > grid + 6)
- Train a deep neural network with time-aware k-fold cross-validation
- Evaluate on validation and test sets
- Display training metrics and feature importances
- Save the trained model to `models/f1_predictor_model_top10.pth`
- Save scaler to `models/scaler_top10.pkl`
- Save results to `json/training_results_top10.json`
- Generate training history plots

**Training Details**:
- Architecture: Input (9 features) → Hidden [128, 64, 32] → Output (1 regression value)
- Optimizer: Adam (learning rate 0.005)
- Loss: PositionAwareLoss (Huber-based with exact position penalty)
- Regularization: Dropout (0.4), Batch Normalization, Weight Decay (2e-4)
- Training: 300 epochs with early stopping (patience 50)
- Validation: Time-aware k-fold CV (5 folds, one per year: 2020-2024)

#### Step 3: Make Predictions

Use the trained model to predict race positions. There are two ways to run predictions:

**Option A: Interactive Mode (Recommended for first-time users)**

Simply run the prediction script without arguments:

```bash
python top10/predict.py
```

This will:
- Load the trained model and test data
- Prompt you to select a year from available options
- Prompt you to select a specific race or type "all" (for 2025) to predict all races
- Display predictions and save results to `predictions.csv`

**Option B: Command Line Arguments**

Provide an input CSV file with driver features:

```bash
python top10/predict.py --input-file race_drivers.csv --output-file predictions.csv
```

**Input CSV Format** (`race_drivers.csv`):

The input CSV should have **one row per driver** competing in the race, with the following columns:

**Required columns (9 features)**:
- `SeasonPoints`: Driver's total season points (float)
- `SeasonStanding`: Driver's championship position (integer, 1 = leader)
- `SeasonAvgFinish`: Driver's average finish position this season (float, lower is better)
- `HistoricalTrackAvgPosition`: Driver's historical average at this track (float, lower is better)
- `ConstructorStanding`: Constructor's championship position (integer, 1 = best)
- `ConstructorTrackAvg`: Constructor's average finish at this track (float, lower is better)
- `GridPosition`: Starting grid position from qualifying (integer, 1-20)
- `RecentForm`: Average finish position in last 3 races (float, lower is better)
- `TrackType`: Binary indicator (1 = street circuit, 0 = permanent circuit)

**Optional columns** (for display purposes):
- `DriverNumber`: Driver number
- `DriverName`: Driver name

**Example input file** (`race_drivers.csv`):
```csv
DriverNumber,DriverName,SeasonPoints,SeasonStanding,SeasonAvgFinish,HistoricalTrackAvgPosition,ConstructorStanding,ConstructorTrackAvg,GridPosition,RecentForm,TrackType
44,HAM,150,2,3.5,4.2,1,2.8,3,3.0,0
33,VER,180,1,2.1,3.8,1,2.8,1,2.3,0
...
```

The prediction script will:
1. Load the trained model and scaler
2. Predict finishing position for each driver
3. Rank all drivers by predicted position (lower score = better rank)
4. Display the **top 10 predicted finishers**
5. Save results to CSV file

## Project Structure

```
.
├── collect_data.py              # Data collection script (Fast F1 API)
├── top10/                       # Top-10 focused model (main model)
│   ├── train.py                 # Training script
│   ├── predict.py               # Prediction script
│   ├── config.py                # Configuration (features, years)
│   ├── model_loader.py          # Model loading utilities
│   └── feature_calculation.py  # Feature engineering
├── top20/                       # Full grid model (baseline)
│   ├── train.py
│   └── predict.py
├── top10_classification/        # Classification variant
├── evaluation/                  # Evaluation scripts
│   ├── evaluate_top10_metrics.py
│   ├── evaluate_baseline_comparison.py
│   └── generate_test_scatter_plot.py
├── data/                        # Data files
│   ├── training_data.csv        # Training data (2020-2024)
│   ├── test_data.csv            # Test data (2025)
│   └── metadata.json            # Data metadata
├── models/                      # Trained models
│   ├── f1_predictor_model_top10.pth
│   ├── scaler_top10.pkl
│   └── ...
├── json/                        # Training results and metrics
│   └── training_results_top10.json
├── images/                      # Generated plots and figures
│   └── prediction_scatter_top10.png
├── cache/                       # Fast F1 API cache
├── requirements.txt             # Python dependencies
├── report.tex                   # Research paper (LaTeX)
└── README.md                    # This file
```

## Feature Engineering

The model uses 9 complementary features:

1. **SeasonPoints**: Total points accumulated in the season
2. **SeasonStanding**: Driver's championship position (1 = leader)
3. **SeasonAvgFinish**: Average finishing position this season
4. **HistoricalTrackAvgPosition**: Driver's historical average at the specific track
5. **ConstructorStanding**: Constructor's championship position
6. **ConstructorTrackAvg**: Constructor's average finish at this specific track
7. **GridPosition**: Starting grid position from qualifying
8. **RecentForm**: Average finish position in last 3 races (captures momentum)
9. **TrackType**: Binary indicator (street circuit vs. permanent circuit)

All features are normalized using StandardScaler (zero mean, unit variance) fitted on training data.

## Model Architecture

**Type**: Deep Feedforward Neural Network (Multi-Layer Perceptron)

**Architecture**:
- Input layer: 9 neurons (9 features)
- Hidden layers: 128 → 64 → 32 neurons
- Output layer: 1 neuron (regression, predicts finishing position)
- Activation: ReLU
- Regularization:
  - Dropout (0.4) after each hidden layer
  - Batch Normalization after each hidden layer
  - Weight Decay (L2 regularization, 2e-4)
- Weight initialization: He/Kaiming initialization
- Total parameters: ~44,161

**Loss Function**: PositionAwareLoss (Huber loss with additional penalty for exact position misses)

**Training**:
- Optimizer: Adam (learning rate 0.005)
- Learning rate scheduling: ReduceLROnPlateau
- Batch size: 32
- Epochs: 300 with early stopping (patience 50)
- Validation: Time-aware k-fold cross-validation (5 folds, one per year)

## Data Collection Details

The data collection script (`collect_data.py`) pulls data from the Fast F1 API and processes it into features:

**Training Data**: 2020-2024 seasons (configured in `top10/config.py`)
**Test Data**: 2025 season

**Data Processing**:
- Calculates all 9 features for each driver-race combination
- Ensures temporal ordering (no future data leakage)
- Handles missing values and edge cases
- Filters DNFs and extreme outliers during training

**Caching**: Fast F1 uses local caching in the `cache/` directory. First run downloads data from API, subsequent runs use cached data (much faster).

## Evaluation

The model is evaluated using:

- **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual positions
- **Root Mean Squared Error (RMSE)**: Penalizes larger errors more heavily
- **R² Score**: Coefficient of determination
- **Position Accuracy**: Percentage of predictions within 1, 2, or 3 positions of actual

Evaluation results are saved to `json/training_results_top10.json` and include:
- Cross-validation metrics (average across 5 folds)
- Final model validation metrics (2024)
- Test set metrics (2025, filtered and unfiltered)

## Output Files

After training, the following files are generated:

- `models/f1_predictor_model_top10.pth`: Trained PyTorch model
- `models/scaler_top10.pkl`: Feature scaler (StandardScaler)
- `models/label_encoder_top10.pkl`: Label encoder (if used)
- `json/training_results_top10.json`: Detailed training metrics and evaluation results
- `images/training_history_top10.png`: Training/validation loss curves
- `images/prediction_scatter_top10.png`: Predicted vs actual positions scatter plot

## Troubleshooting

### Data Collection Issues

**Problem**: `collect_data.py` fails with API errors
- **Solution**: The script includes retry logic. If errors persist, check your internet connection and try again later. Fast F1 API may be temporarily unavailable.

**Problem**: Data collection is slow
- **Solution**: First run downloads data from API. Subsequent runs use cached data and are much faster. The cache is stored in `cache/` directory.

### Training Issues

**Problem**: `FileNotFoundError: Training data not found`
- **Solution**: Run `collect_data.py` first, or ensure `data/training_data.csv` exists.

**Problem**: CUDA/GPU errors
- **Solution**: The code automatically uses CPU if CUDA is not available. Training works on CPU but is slower.

### Prediction Issues

**Problem**: Model file not found
- **Solution**: Train the model first using `python top10/train.py`

**Problem**: Feature mismatch errors
- **Solution**: Ensure your input CSV contains all 9 required features with correct column names (case-sensitive).

## Configuration

Model configuration is centralized in `top10/config.py`. You can modify the following settings:

- `FEATURE_COLS`: List of 9 feature column names
- `TRAINING_YEARS`: Years to use for training data (default: [2020, 2021, 2022, 2023, 2024])
- `TEST_YEAR`: Year to use for test data (default: 2025)

**To change training/test years**: Edit `top10/config.py` and modify the `TRAINING_YEARS` and `TEST_YEAR` variables. After modifying `config.py`, you'll need to:
1. Re-run `collect_data.py` to collect data for the new years (if needed)
2. Re-train the model with `python top10/train.py` to use the new configuration

**Note**: The training years and test year are configured in `top10/config.py`. You can modify `TRAINING_YEARS` (list of years) and `TEST_YEAR` (single year) to change which seasons are used for training and testing.

## Dependencies

See `requirements.txt` for full list. Key dependencies:

- `fastf1>=3.1.0`: F1 data API
- `pandas>=2.0.0`: Data manipulation
- `numpy>=1.24.0`: Numerical computing
- `torch>=2.0.0`: Deep learning framework
- `scikit-learn>=1.3.0`: Machine learning utilities
- `matplotlib>=3.7.0`: Plotting

## License

[Add your license information here]

## Citation

If you use this code in your research, please cite:

[Add citation information here]

## Contact

[Add contact information here]
