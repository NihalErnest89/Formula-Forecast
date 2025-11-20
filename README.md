# F1 Predictions

Predict F1 race finishing positions using deep learning neural networks based on driver performance features.

## Overview

This project uses **neural networks** (deep learning) to predict F1 race finishing positions and ranks drivers to show the **predicted top 10 finishers**. The model is trained on historical F1 data and uses 7 carefully engineered features to capture driver performance, track history, and current form.

### Base Features (7 features)
- **Season Points**: Total points accumulated in the season
- **Season Average Finish Position**: Average finishing position this season
- **Historical Track Average Position**: Driver's historical average position at the specific track
- **Constructor Standing**: Constructor's championship position
- **Grid Position**: Average grid start position
- **Recent Form**: Average finish position in last 5 races (captures current momentum)
- **Track Type**: Binary indicator (street circuit vs. permanent circuit)


## Project Structure

- `collect_data.py`: Script to pull F1 data from Fast F1 library and organize it into features/labels
- `top10/`: Directory for top 10 finisher models (positions 1-10)
  - `train.py`: Train neural network on top 10 finishers only
  - `predict.py`: Predict top 10 finishers using the top 10 model
- `top20/`: Directory for full grid models (positions 1-20)
  - `train.py`: Train neural network on all 20 positions
  - `predict.py`: Predict all 20 positions using the full grid model
- `train.py`: Legacy training script (all 20 positions)
- `train_rf.py`: Script to train a **Random Forest** (traditional ML approach for comparison)
- `predict.py`: Legacy prediction script (all 20 positions)
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
   # Train Top 10 Model (Recommended - focuses on competitive finishers)
   python top10/train.py
   
   # Train Full Grid Model (All 20 positions)
   python top20/train.py
   
   # Train Random Forest (Traditional ML - for comparison)
   python train_rf.py
   ```

4. **Step 3**: Make predictions
   ```bash
   # Using Top 10 Model (Recommended - focuses on competitive finishers)
   python top10/predict.py --input-file race_drivers.csv
   
   # Using Full Grid Model (All 20 positions)
   python top20/predict.py --input-file race_drivers.csv
   
   # Using Random Forest
   python predict.py --model-type random_forest --input-file race_drivers.csv
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
- Collect data from 2022-2024 seasons for training
- Collect data from 2025 season for testing
- Calculate features (Season Points, Season Avg Finish, Historical Track Avg Position)
- Save data to `data/training_data.csv` and `data/test_data.csv`

### Step 2: Train Model(s)

You can train either a Neural Network (deep learning) or Random Forest (traditional ML), or both for comparison:

#### Option A: Top 10 Model (Recommended)

```bash
python top10/train.py
```

This will:
- Load training and test data
- Filter to top 10 finishers only (positions 1-10)
- Train a deep neural network with multiple hidden layers
- Use PyTorch with ReLU activations, dropout, and batch normalization
- Train for multiple epochs with early stopping
- Evaluate on validation and test sets
- Display feature importances (from first layer weights)
- Save the trained model to `models/f1_predictor_model_top10.pth`
- Save results to `training_results_top10.json`
- Generate training history plots

**Architecture:**
- Input: 7 features
- Hidden layers: 256 → 128 → 64 neurons
- Activation: ReLU
- Regularization: Dropout (0.4) and Batch Normalization
- Output: Single regression value (predicted finishing position 1-10)
- Training: Only on drivers who finished in positions 1-10

#### Option B: Full Grid Model (All 20 Positions)

```bash
python top20/train.py
```

This will:
- Load training and test data
- Train on all 20 positions
- Train a deep neural network with multiple hidden layers
- Use PyTorch with ReLU activations, dropout, and batch normalization
- Train for multiple epochs with early stopping
- Evaluate on validation and test sets
- Display feature importances (from first layer weights)
- Save the trained model to `models/f1_predictor_model.pth`
- Save results to `training_results.json`
- Generate training history plots

**Architecture:**
- Input: 7 features
- Hidden layers: 192 → 96 → 48 neurons
- Activation: ReLU
- Regularization: Dropout (0.4) and Batch Normalization
- Output: Single regression value (predicted finishing position 1-20)

#### Option C: Random Forest (Traditional ML)

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
# Top 10 Model (Recommended - focuses on competitive finishers)
python top10/predict.py --input-file race_drivers.csv --output-file top10_predictions.csv

# Full Grid Model (All 20 positions)
python top20/predict.py --input-file race_drivers.csv --output-file top20_predictions.csv

# Random Forest (Legacy)
python predict.py --model-type random_forest --input-file race_drivers.csv --output-file top10_predictions.csv
```

**Input CSV format:**
The input CSV should have **one row per driver** competing in the race, with columns:

**Required columns (base features):**
- `SeasonPoints`: Driver's season points
- `SeasonAvgFinish`: Driver's average finish position this season
- `HistoricalTrackAvgPosition`: Driver's historical average at this track
- `ConstructorStanding`: Constructor's championship standing (1 = best)
- `GridPosition`: Starting grid position (qualifying position) or average grid position
- `RecentForm`: Average finish position in last 5 races
- `TrackType`: Binary (1 = street circuit, 0 = permanent circuit)

**Optional columns:**
- `DriverNumber`: Driver number (for display)
- `DriverName`: Driver name (for display)


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

**Base Features (Required):**
- `SeasonPoints`: Float value
- `SeasonAvgFinish`: Float value (lower is better)
- `HistoricalTrackAvgPosition`: Float value (lower is better)
- `ConstructorStanding`: Integer (1 = best constructor, higher = worse)
- `GridPosition`: Integer (1-20, starting grid position from qualifying)
- `RecentForm`: Float value (average finish in last 5 races, lower is better)
- `TrackType`: Integer (1 = street circuit, 0 = permanent circuit)


## Model Details

### Neural Network (Deep Learning)

We used a **Multi-Layer Perceptron (MLP) neural network for regression** to predict F1 race finishing positions. Here's how we implemented it:

#### 1. Problem Formulation
- **Task**: Regression (predict finishing position 1–10 for top 10 model, 1–20 for full grid model)
- **Input**: 
  - **Top 10 Model**: 7 features per driver
  - **Full Grid Model**: 7 features per driver
- **Output**: Single continuous value (predicted finishing position)

#### 2. Architecture

**Top 10 Model:**
- **Type**: Feedforward neural network (MLP)
- **Structure**:
  - Input layer: 7 neurons (7 features)
  - Hidden layers: 256 → 128 → 64 neurons
  - Output layer: 1 neuron (regression)
- **Activation**: ReLU (Rectified Linear Unit)
- **Regularization**:
  - Dropout (0.4) to reduce overfitting
  - Batch normalization for stable training
  - Weight decay (L2 regularization, 2e-4)

**Full Grid Model:**
- **Type**: Feedforward neural network (MLP)
- **Structure**:
  - Input layer: 7 neurons (base features)
  - Hidden layers: 192 → 96 → 48 neurons
  - Output layer: 1 neuron (regression)
- **Activation**: ReLU (Rectified Linear Unit)
- **Regularization**:
  - Dropout (0.4) to reduce overfitting
  - Batch normalization for stable training
  - Weight decay (L2 regularization, 2e-4)

#### 3. Training
- **Framework**: PyTorch
- **Loss Function**: Huber loss (robust to outliers like DNFs/crashes)
- **Optimizer**: Adam
- **Learning Rate**: 0.001 with ReduceLROnPlateau scheduler
- **Early Stopping**: Patience of 20 epochs
- **Data**: 2022–2024 seasons (training), 2025 (test)
- **Preprocessing**: StandardScaler (zero mean, unit variance)

#### 4. Design Decisions
- **Equal Weight Initialization**: First layer initialized to give equal importance to all features (~14.29% each)
- **Gradient Clipping**: Max norm 1.0 to prevent exploding gradients
- **Outlier Handling**: Huber loss and outlier clipping in preprocessing (3 standard deviations)
- **Best Model Checkpointing**: Saves the best validation model during training

#### 5. Results

**Top 10 Model:**
- **Test MAE**: ~1.89 positions (on top 10 finishers)
- **Within 1 position**: ~50-60% (varies by race)
- **Within 2 positions**: ~70-80%
- **Within 3 positions**: ~80-90%
- **Exact winner prediction**: Strong performance on 2025 season races

**Full Grid Model:**
- **Test MAE**: ~3.13 positions (on all 20 positions)
- **Within 1 position**: ~23%
- **Within 3 positions**: ~61%
- **R²**: ~0.46

#### 6. Why a Neural Network?
- **Non-linear Relationships**: Captures complex interactions between features (e.g., how RecentForm interacts with TrackType)
- **Feature Learning**: Learns which features matter most for different scenarios
- **Scalability**: Easy to add features or adjust architecture
- **Pattern Recognition**: Identifies subtle patterns in driver performance that linear models miss
- **Top 10 Specialization**: Training only on top 10 finishers allows the model to focus on competitive drivers

The model learns to weight features appropriately to predict where each driver will finish, accounting for non-linear relationships between driver performance, track history, recent form, and constructor competitiveness. The top 10 model specializes in predicting competitive finishers by training exclusively on top 10 data.


## Output

- `data/`: Contains collected training and test data
- `models/`: Contains trained models and scalers
  - `f1_predictor_model_top10.pth`: Top 10 neural network model (PyTorch, 7 features)
  - `scaler_top10.pkl`: Feature scaler for top 10 model
  - `f1_predictor_model.pth`: Full grid neural network model (PyTorch, 7 features)
  - `scaler.pkl`: Feature scaler for full grid model
  - `f1_predictor_model_rf.pkl`: Random Forest model
  - `scaler_rf.pkl`: Feature scaler for Random Forest
- `training_results_top10.json`: Top 10 model training metrics and feature importances
- `training_results.json`: Full grid model training metrics and feature importances
- `training_results_rf.json`: Random Forest training metrics and feature importances
- `training_history.png`: Training/validation MAE curves (neural network)
- `prediction_scatter.png`: Predicted vs actual positions scatter plot (neural network)
- `prediction_scatter_rf.png`: Predicted vs actual positions scatter plot (Random Forest)

