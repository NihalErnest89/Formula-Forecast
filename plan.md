# F1 Predictions

## Features
- Season Points
- Season Average Finish Position
- Historical Track Average Position
- Constructor Points
- Constructor Standing
- Grid Position (Qualifying Position) - Starting position on the grid
- Recent Form - Average finish position in last 5 races (current momentum)

## Labels
- Race Finishing Position (1-20) - Predict the position each driver will finish in a race

## Data
- Fast F1
- Training:
  - Past 5 seasons of data
- Test:
  - Data from current 2025 season

## Objectives
- Predict race finishing positions (1-20) for each driver
- Rank drivers to show predicted top 10 for a race
- Identify the correct features:
  - Start with the 7 listed above (Season Points, Season Average Finish Position, Historical Track Average Position, Constructor Points, Constructor Standing, Grid Position, Recent Form)
  - Use learning to get the correct weight distribution for features
  - Achieve good accuracy with race position predictions

