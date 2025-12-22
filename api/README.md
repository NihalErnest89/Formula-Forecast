# Formula Forecast - API Setup

Flask REST API for F1 Race Position Predictions.

## Prerequisites

- Python 3.8 or higher
- Trained model files in `../models/` directory:
  - `f1_predictor_model_top10.pth` (or similar)
  - `scaler_top10.pkl`
- Data files in `../data/` directory:
  - `test_data.csv`
  - `training_data.csv` (optional, for future race predictions)

## Installation

1. Install dependencies from the main requirements.txt:
```bash
pip install -r ../requirements.txt
```

This will install all dependencies including Flask and flask-cors.

## Running the API

1. Make sure you're in the `api` directory:
```bash
cd api
```

2. Start the Flask server:
```bash
python app.py
```

The API will run on `http://localhost:5000`

## API Endpoints

### Health Check
```
GET /api/health
```
Returns the health status of the API and whether the model/data are loaded.

### Get Available Races
```
GET /api/races
```
Returns a list of all available races (completed and future races).

Response:
```json
{
  "races": [
    {
      "year": 2025,
      "eventName": "Sao Paulo Grand Prix",
      "roundNumber": 1,
      "isFuture": false
    }
  ]
}
```

### Make Predictions
```
POST /api/predict
Content-Type: application/json

{
  "year": 2025,
  "eventName": "Sao Paulo Grand Prix",
  "roundNumber": 1
}
```

Returns predictions for the top 10 drivers:
```json
{
  "race": {
    "year": 2025,
    "eventName": "Sao Paulo Grand Prix",
    "roundNumber": 1,
    "isFuture": false
  },
  "predictions": [
    {
      "rank": 1,
      "driverName": "Max Verstappen",
      "driverNumber": 1,
      "predictedPosition": 1.23,
      "gridPosition": 1,
      "constructor": "Red Bull Racing"
    }
  ],
  "totalDrivers": 20
}
```

## Troubleshooting

- **Model not loaded**: Make sure model files exist in `../models/` directory
- **Data not loaded**: Ensure `test_data.csv` exists in `../data/` directory
- **CORS errors**: The API includes CORS headers, but make sure the frontend is configured correctly
- **Port already in use**: Change the port in `app.py` (last line)

