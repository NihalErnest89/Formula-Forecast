# F1 Predictions

Predict F1 race finishing positions using deep learning neural networks based on driver performance features.

## Overview

This project uses **neural networks** (deep learning) to predict F1 race finishing positions and ranks drivers to show the **predicted top 10 finishers**. The model is trained on historical F1 data and uses 7 carefully engineered features to capture driver performance, track history, and current form.

### Base Features (9 features)
- **Season Points**: Total points accumulated in the season
- **Season Standing**: Driver's championship position (1 = leader, higher = worse)
- **Season Average Finish Position**: Average finishing position this season
- **Historical Track Average Position**: Driver's historical average position at the specific track
- **Constructor Standing**: Constructor's championship position
- **Constructor Track Average**: Constructor's average finish at this specific track
- **Grid Position**: Average grid start position (season-specific average)
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
- Input: 9 features
- Hidden layers: 128 → 64 → 32 neurons
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
- Input: 9 features
- Hidden layers: 128 → 64 → 32 neurons
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
- `SeasonStanding`: Driver's championship position (1 = leader, higher = worse)
- `SeasonAvgFinish`: Driver's average finish position this season
- `HistoricalTrackAvgPosition`: Driver's historical average at this track
- `ConstructorStanding`: Constructor's championship standing (1 = best)
- `ConstructorTrackAvg`: Constructor's average finish at this specific track
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
- `SeasonStanding`: Integer (1 = leader, higher = worse)
- `SeasonAvgFinish`: Float value (lower is better)
- `HistoricalTrackAvgPosition`: Float value (lower is better)
- `ConstructorStanding`: Integer (1 = best constructor, higher = worse)
- `ConstructorTrackAvg`: Float value (constructor's average finish at this track, lower is better)
- `GridPosition`: Integer (1-20, starting grid position from qualifying or average)
- `RecentForm`: Float value (average finish in last 5 races, lower is better)
- `TrackType`: Integer (1 = street circuit, 0 = permanent circuit)


## Model Details

### Neural Network (Deep Learning)

We used a **Multi-Layer Perceptron (MLP) neural network for regression** to predict F1 race finishing positions. Here's how we implemented it:

#### 1. Problem Formulation
- **Task**: Regression (predict finishing position 1–10 for top 10 model, 1–20 for full grid model)
- **Input**: 
  - **Top 10 Model**: 9 features per driver
  - **Full Grid Model**: 9 features per driver
- **Output**: Single continuous value (predicted finishing position)

#### 2. Architecture

**Top 10 Model:**
- **Type**: Feedforward neural network (MLP)
- **Structure**:
  - Input layer: 9 neurons (9 features)
  - Hidden layers: 128 → 64 → 32 neurons
  - Output layer: 1 neuron (regression)
- **Activation**: ReLU (Rectified Linear Unit)
- **Regularization**:
  - Dropout (0.4) to reduce overfitting
  - Batch normalization for stable training
  - Weight decay (L2 regularization, 2e-4)

**Full Grid Model:**
- **Type**: Feedforward neural network (MLP)
- **Structure**:
  - Input layer: 9 neurons (9 features)
  - Hidden layers: 128 → 64 → 32 neurons
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
- **Early Stopping**: Patience of 50 epochs
- **Data**: 2020–2024 seasons (training), 2025 (test)
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
  - `f1_predictor_model_top10.pth`: Top 10 neural network model (PyTorch, 9 features)
  - `scaler_top10.pkl`: Feature scaler for top 10 model
  - `f1_predictor_model.pth`: Full grid neural network model (PyTorch, 9 features)
  - `scaler.pkl`: Feature scaler for full grid model
  - `f1_predictor_model_rf.pkl`: Random Forest model
  - `scaler_rf.pkl`: Feature scaler for Random Forest
- `training_results_top10.json`: Top 10 model training metrics and feature importances
- `training_results.json`: Full grid model training metrics and feature importances
- `training_results_rf.json`: Random Forest training metrics and feature importances
- `training_history.png`: Training/validation MAE curves (neural network)
- `prediction_scatter.png`: Predicted vs actual positions scatter plot (neural network)
- `prediction_scatter_rf.png`: Predicted vs actual positions scatter plot (Random Forest)

## Experimentation History

This section documents experiments we've tried, what worked, and what didn't.

### ✅ Implemented and Working

#### 1. **Dataset Expansion (2020-2024)**
- **What**: Expanded training data from 2022-2024 to 2020-2024
- **Result**: ✅ **Improved performance** - Better generalization, more historical context
- **Status**: Currently in use
- **Impact**: More data points for driver-track combinations and constructor performance trends

#### 2. **Batch Normalization**
- **What**: Added batch normalization layers to stabilize training
- **Result**: ✅ **Improved performance** - Beneficial even with small batch sizes (32)
- **Status**: Currently in use
- **Note**: Initially tested without batch norm due to concerns about small batch sizes, but results showed it helps

#### 3. **K-Fold Cross-Validation Strategy (1 fold per year)**
- **What**: Time-aware k-fold validation (one fold per year: 2020, 2021, 2022, 2023, 2024)
- **Result**: ✅ **Optimal** - Tested 3 folds per year (splitting each year into thirds), but minimal gains didn't justify increased runtime
- **Status**: Currently using 1 fold per year (5 folds total for 2020-2024)

#### 4. **Early Stopping**
- **What**: Added early stopping with patience of 50 epochs
- **Result**: ✅ **Working** - Prevents overfitting and saves computation time
- **Status**: Currently in use

#### 5. **Feature Set (9 features)**
- **What**: Current feature set includes:
  - SeasonPoints, SeasonStanding, SeasonAvgFinish
  - HistoricalTrackAvgPosition, ConstructorStanding, ConstructorTrackAvg
  - GridPosition, RecentForm, TrackType
- **Result**: ✅ **Working well** - Balanced feature set with good predictive power
- **Status**: Currently in use
- **Note**: Expanded from 7 to 9 features (added SeasonStanding and ConstructorTrackAvg)

### ❌ Tested and Rejected

#### 1. **ConstructorRecentForm Feature**
- **What**: Added constructor's recent form (average finish of both drivers in last 5 races) as a new feature
- **Result**: ❌ **Worse performance** - Actually decreased accuracy
- **Reason**: Redundant with existing features:
  - ConstructorStanding (championship position)
  - ConstructorTrackAvg (track-specific performance)
  - RecentForm (driver-specific, which already reflects car performance)
- **Status**: Removed, not in current model

#### 2. **Quick Wins (LR Warmup, GELU, Gradient Clipping)**
- **What**: Tested learning rate warmup, GELU activation, and gradient clipping individually and combined
- **Result**: ❌ **No improvement** - Baseline configuration was already optimal
- **Status**: Not implemented
- **Note**: Gradient clipping is still used (max norm 1.0) as it's a standard practice, but adding it didn't improve results

### 🔄 In Progress / Under Testing

#### 1. **Feature Normalization/Scaling Improvements**
- **What**: Improved feature-specific normalization:
  - SeasonPoints: Log transform + percentile rank (0-1 scale)
  - GridPosition: Inverse transform (20 = best, 1 = worst) then normalize
  - Position features: Ensure consistent 0-1 scaling
- **Status**: 🔄 Testing in `top10/compare_feature_improvements.py`
- **Hypothesis**: Better feature scaling will help model learn more effectively, especially for features with different scales

#### 2. **Attention Mechanism**
- **What**: Added attention mechanism to dynamically weight features based on context
- **Status**: 🔄 Testing in `top10/compare_feature_improvements.py`
- **Hypothesis**: Different features may be more important for different drivers/races (e.g., track-specific features more important at certain tracks)

#### 3. **Feature Combining**
- **What**: Combining similar features into composite features:
  - ChampionshipStrength: SeasonPoints + SeasonStanding (60% points, 40% standing)
  - TrackPerformance: HistoricalTrackAvgPosition + ConstructorTrackAvg (50/50)
  - FormStrength: RecentForm + SeasonAvgFinish (70% recent, 30% average)
- **Status**: 🔄 Testing in `top10/compare_feature_improvements.py`
- **Hypothesis**: Reducing feature count while preserving information may improve model focus

### 📊 Feature Importance Analysis

Based on weight progression during training, current feature importances (normalized):
- **GridPosition**: ~15.5-16% (highest - most important)
- **ConstructorStanding**: ~11.5-12%
- **RecentForm**: ~11.4%
- **HistoricalTrackAvgPosition**: ~11.1%
- **ConstructorTrackAvg**: ~12.3%
- **SeasonAvgFinish**: ~10.4%
- **SeasonPoints**: ~10.0%
- **SeasonStanding**: ~9.7%
- **TrackType**: ~9.8% (lowest)

**Observations**:
- GridPosition is overrepresented (may need rebalancing)
- SeasonPoints + SeasonStanding combined (~20%) may be overrepresented
- RecentForm and ConstructorTrackAvg may be underrepresented given their predictive value
- These observations led to the current experiments in feature normalization and attention mechanisms

### 🎯 Future Experiments to Consider

1. **Recent Performance Features**: Replace cumulative stats (SeasonPoints) with recent performance (last 3-5 races)
2. **Momentum Mismatch Feature**: Flag drivers with high cumulative stats but poor recent form
3. **Feature-Specific Regularization**: Penalize over-reliance on certain features (e.g., SeasonPoints)
4. **Separate Feature Pathways**: Different neural pathways for recent performance, track-specific, and baseline stats
5. **Track-Specific Attention**: Increase track-specific feature weights for track-specific races

### 📝 Notes on Experimentation Process

- All experiments use time-aware k-fold cross-validation (one fold per year) to prevent data leakage
- Comparisons are made against baseline using same validation strategy
- Feature importance is tracked through first-layer weight analysis
- Results are saved to JSON files for reproducibility
- Scripts for comparisons are kept in `top10/` directory (e.g., `compare_feature_improvements.py`)

