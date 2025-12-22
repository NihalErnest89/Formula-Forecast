# Formula Forecast - Frontend Setup

This directory contains the React frontend for the F1 Race Prediction application.

## Prerequisites

- Node.js (v14 or higher)
- npm or yarn
- Backend API running on port 5000 (see `api/` directory)

## Installation

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

## Running the Application

1. Make sure the backend API is running (see `api/README.md`)

2. Start the React development server:
```bash
npm start
```

The app will open in your browser at `http://localhost:3000`

## Building for Production

To create a production build:

```bash
npm run build
```

The build folder will contain the optimized production build.

## Environment Variables

You can set the API URL by creating a `.env` file in the frontend directory:

```
REACT_APP_API_URL=http://localhost:5000
```

## Features

- Browse available F1 races by year
- Select a race to see predictions
- View top 10 predicted finishing positions
- Responsive design for mobile and desktop
- Real-time predictions using the trained neural network model

