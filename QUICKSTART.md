# Formula Forecast - Quick Start Guide

This guide will help you set up and run the F1 Race Prediction web application.

## Project Structure

```
Need-For-Predictions/
├── api/                 # Flask backend API
│   └── app.py          # Main API application
├── frontend/           # React frontend
│   ├── src/
│   │   ├── App.js      # Main React component
│   │   └── ...
│   └── package.json
├── top10/              # Prediction model code
├── models/             # Trained model files
├── data/               # Training and test data
└── requirements.txt    # Python dependencies (shared)
```

## Prerequisites

1. **Python 3.8+** with pip
2. **Node.js 14+** with npm
3. **Trained model files** in `models/` directory:
   - `f1_predictor_model_top10.pth`
   - `scaler_top10.pkl`
4. **Data files** in `data/` directory:
   - `test_data.csv`
   - `training_data.csv` (optional)

## Setup Steps

### 1. Backend Setup

1. Install Python dependencies from the root directory:
```bash
pip install -r requirements.txt
```

This installs all dependencies including Flask and flask-cors for the API.

### 2. Frontend Setup

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install Node.js dependencies:
```bash
npm install
```

### 3. Running the Application

**Terminal 1 - Start the Backend API:**
```bash
cd api
python app.py
```

The API will start on `http://localhost:5000`

**Terminal 2 - Start the Frontend:**
```bash
cd frontend
npm start
```

The React app will open automatically at `http://localhost:3000`

## Usage

1. Open your browser to `http://localhost:3000`
2. Select a year from the dropdown (or view all years)
3. Click on a race to see predictions
4. View the top 10 predicted finishing positions

## Troubleshooting

### Backend Issues

- **Model not found**: Make sure model files exist in `models/` directory. Train the model first:
  ```bash
  python top10/train.py
  ```

- **Data not found**: Ensure `data/test_data.csv` exists. Generate it with:
  ```bash
  python collect_data.py
  ```

- **Port 5000 already in use**: Edit `api/app.py` and change the port number in the last line.

### Frontend Issues

- **Cannot connect to API**: Make sure the backend is running on port 5000
- **npm install fails**: Try deleting `node_modules` and `package-lock.json`, then run `npm install` again
- **CORS errors**: The backend includes CORS headers. If issues persist, check that the API URL is correct

## API Endpoints

- `GET /api/health` - Health check
- `GET /api/races` - Get list of available races
- `POST /api/predict` - Get predictions for a race

See `api/README.md` for detailed API documentation.

## Development

### Backend Development
- The Flask app runs in debug mode by default
- Changes to `api/app.py` require restarting the server

### Frontend Development
- React hot-reloads automatically
- Changes to `frontend/src/` files will update in the browser automatically

## Production Build

To build the frontend for production:

```bash
cd frontend
npm run build
```

The optimized build will be in `frontend/build/`.

For production deployment:
- Use a production WSGI server like Gunicorn for Flask
- Serve the React build with a web server like Nginx
- Configure environment variables appropriately

