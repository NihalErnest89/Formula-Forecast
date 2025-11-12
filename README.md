# F1 Predictions

Predict F1 race winners using machine learning based on driver performance features.

## Overview

This project predicts F1 race finishing positions and ranks drivers to show the **predicted top 10** using six key features:
- **Season Points**: Total points accumulated in the season
- **Season Average Finish Position**: Average finishing position this season
- **Historical Track Average Position**: Driver's historical average position at the specific track
- **Constructor Points**: Constructor's total points
- **Constructor Standing**: Constructor's championship position
- **Grid Position**: Starting grid position (qualifying position)

## Project Structure

- `collect_data.py`: Script to pull F1 data from Fast F1 library and organize it into features/labels
- `train.py`: Script to train a **Neural Network** (deep learning approach)
- `train_rf.py`: Script to train a **Random Forest** (traditional ML approach for comparison)
- `predict.py`: Script to run either trained model with given input features
- `requirements.txt`: Python dependencies
- `plan.md`: Project plan and objectives

## Quick Start (Execution Order)

Run these scripts in order:

1. **First time setup**: Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

2. **Step 1**: Collect data
   ```bash
   python collect_data.py
   ```

3. **Step 2**: Train the model(s)
   ```bash
   # Train Neural Network (Deep Learning)
   python train.py
   
   # Train Random Forest (Traditional ML - for comparison)
   python train_rf.py
   ```

4. **Step 3**: Make predictions
   ```bash
   # Using Neural Network (default)
   python predict.py --season-points 150 --season-avg-finish 5.2 --historical-track-avg 4.8
   
   # Using Random Forest
   python predict.py --model-type random_forest --season-points 150 --season-avg-finish 5.2 --historical-track-avg 4.8
   ```

**Note**: After the first run, you only need to run steps 2 and 3 again if you want to retrain or make new predictions. The data collection (step 1) uses caching, so subsequent runs are much faster.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Fast F1 uses caching. The cache directory will be created automatically when you run `collect_data.py`.
   - **First run**: Will download data from Fast F1 API (may take some time)
   - **Subsequent runs**: Will use cached data (much faster, no API calls)
   - Cache is stored in the `cache/` directory

## Usage

### Step 1: Collect Data

Run the data collection script to pull F1 data from Fast F1:

```bash
python collect_data.py
```

This will:
- Collect data from past 5 seasons (2020-2024) for training
- Collect data from 2025 season for testing
- Calculate features (Season Points, Season Avg Finish, Historical Track Avg Position)
- Save data to `data/training_data.csv` and `data/test_data.csv`

### Step 2: Train Model(s)

You can train either a Neural Network (deep learning) or Random Forest (traditional ML), or both for comparison:

#### Option A: Neural Network (Deep Learning)

```bash
python train.py
```

This will:
- Load training and test data
- Train a deep neural network with multiple hidden layers
- Use PyTorch with ReLU activations, dropout, and batch normalization
- Train for multiple epochs with early stopping
- Evaluate on validation and test sets
- Display feature importances (from first layer weights)
- Save the trained model to `models/f1_predictor_model.pth`
- Save results to `training_results.json`
- Generate training history plots

**Architecture:**
- Input: 3 features
- Hidden layers: 128 → 64 → 32 neurons
- Activation: ReLU
- Regularization: Dropout (0.3) and Batch Normalization
- Output: Softmax over driver classes

#### Option B: Random Forest (Traditional ML)

```bash
python train_rf.py
```

This will:
- Load training and test data
- Train a Random Forest classifier (100 trees)
- Evaluate on validation and test sets
- Display feature importances (from tree splits)
- Save the trained model to `models/f1_predictor_model_rf.pkl`
- Save results to `training_results_rf.json`

**Why compare both?**
- **Neural Network**: Deep learning approach with learnable weights, backpropagation, and non-linear activations
- **Random Forest**: Traditional ensemble method, interpretable, fast training
- Compare accuracy, training time, and feature importance interpretations

### Step 3: Make Predictions

Use the trained model to predict race positions and get the **top 10 finishers**.

**Predict top 10 from CSV file (recommended):**
```bash
# Neural Network (default)
python predict.py --input-file race_drivers.csv --output-file top10_predictions.csv

# Random Forest
python predict.py --model-type random_forest --input-file race_drivers.csv --output-file top10_predictions.csv

# Show all drivers (not just top 10)
python predict.py --input-file race_drivers.csv --show-all
```

**Input CSV format:**
The input CSV should have **one row per driver** competing in the race, with columns:
- `SeasonPoints`: Driver's season points
- `SeasonAvgFinish`: Driver's average finish position this season
- `HistoricalTrackAvgPosition`: Driver's historical average at this track
- `ConstructorPoints`: Constructor's total points
- `ConstructorStanding`: Constructor's championship standing (1 = best)
- `GridPosition`: Starting grid position (qualifying position)
- `DriverNumber`: Driver number (optional, for display)
- `DriverName`: Driver name (optional, for display)

The script will:
1. Predict finishing position for each driver
2. Rank all drivers by predicted position
3. Display the **top 10 finishers**
4. Save results to CSV

## Data Format

### Training/Test Data
The data collection script creates CSV files with the following columns:
- `Year`: Season year
- `EventName`: Track/race name
- `RoundNumber`: Race round number
- `SeasonPoints`: Total season points for the driver
- `SeasonAvgFinish`: Average finish position this season
- `HistoricalTrackAvgPosition`: Historical average position at this track
- `DriverNumber`: Driver number (label)
- `DriverName`: Driver abbreviation
- `ActualPosition`: Actual finishing position

### Input Features for Prediction
When using `predict.py`, provide a CSV with one row per driver containing:
- `SeasonPoints`: Float value
- `SeasonAvgFinish`: Float value (lower is better)
- `HistoricalTrackAvgPosition`: Float value (lower is better)
- `ConstructorPoints`: Float value
- `ConstructorStanding`: Integer (1 = best constructor, higher = worse)
- `GridPosition`: Integer (1-20, starting grid position from qualifying)

## Model Details

### Neural Network (Deep Learning)
- **Framework**: PyTorch
- **Architecture**: Multi-layer Perceptron (MLP)
  - Input: 6 features
  - Hidden: 128 → 64 → 32 neurons
  - Activation: ReLU
  - Regularization: Dropout (0.3), Batch Normalization
  - Output: Single value (predicted finishing position 1-20)
  - **Initialization**: Equal weights for all features (~16.67% each)
- **Training**: Adam optimizer, MSELoss (regression), learning rate scheduling, early stopping
- **Features**: 6 features (Season Points, Season Avg Finish, Historical Track Avg, Constructor Points, Constructor Standing, Grid Position)
- **Label**: Actual finishing position (1-20)
- **Preprocessing**: Standard scaling of features

### Random Forest (Traditional ML)
- **Algorithm**: Random Forest Regressor
- **Parameters**: 100 trees, max depth 10
- **Features**: 6 features (Season Points, Season Avg Finish, Historical Track Avg, Constructor Points, Constructor Standing, Grid Position)
- **Initialization**: Considers all features equally at start, learns optimal importance
- **Label**: Actual finishing position (1-20)
- **Preprocessing**: Standard scaling of features

## Output

- `data/`: Contains collected training and test data
- `models/`: Contains trained models and scalers
  - `f1_predictor_model.pth`: Neural network model (PyTorch)
  - `scaler.pkl`: Feature scaler for neural network
  - `f1_predictor_model_rf.pkl`: Random Forest model
  - `scaler_rf.pkl`: Feature scaler for Random Forest
- `training_results.json`: Neural network training metrics and feature importances
- `training_results_rf.json`: Random Forest training metrics and feature importances
- `training_history.png`: Training/validation MAE curves (neural network)
- `prediction_scatter.png`: Predicted vs actual positions scatter plot (neural network)
- `prediction_scatter_rf.png`: Predicted vs actual positions scatter plot (Random Forest)

