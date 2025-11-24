"""
Data collection script for F1 Predictions.
Pulls data from Fast F1 library and organizes it into features and labels.
"""

import fastf1
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Enable caching for Fast F1
# This will cache all API calls to avoid expensive re-downloads
# First run will download data, subsequent runs will use cached data
cache_dir = Path('cache')
cache_dir.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(cache_dir))
print(f"Fast F1 cache enabled at: {cache_dir.absolute()}")


def get_season_data(year: int, max_retries: int = 3) -> pd.DataFrame:
    """
    Get all race data for a given season.
    
    Args:
        year: Season year (e.g., 2024)
        max_retries: Maximum number of retry attempts for API calls
        
    Returns:
        DataFrame with race results for the season
    """
    schedule = None
    
    # Try to get schedule with retries
    for attempt in range(max_retries):
        try:
            schedule = fastf1.get_event_schedule(year)
            break
        except (ValueError, Exception) as e:
            if attempt < max_retries - 1:
                print(f"  Warning: Failed to load schedule for {year} (attempt {attempt + 1}/{max_retries}). Retrying...")
                import time
                time.sleep(2)  # Wait 2 seconds before retry
            else:
                print(f"  Error: Failed to load schedule for {year} after {max_retries} attempts: {e}")
                print(f"  Skipping season {year}. This may be due to API issues or network problems.")
                return pd.DataFrame()
    
    if schedule is None or schedule.empty:
        print(f"  Warning: No schedule found for {year}")
        return pd.DataFrame()
    
    all_races = []
    
    for _, event in schedule.iterrows():
        try:
            # Get race session
            session = fastf1.get_session(year, event['EventName'], 'R')
            session.load()
            
            # Get race results
            results = session.results
            if results is not None and not results.empty:
                results['Year'] = year
                results['EventName'] = event['EventName']
                results['RoundNumber'] = event['RoundNumber']
                
                # Try to get sprint points if sprint race exists
                sprint_points_dict = {}
                try:
                    sprint_session = fastf1.get_session(year, event['EventName'], 'Sprint')
                    sprint_session.load(telemetry=False, weather=False, messages=False, laps=False)
                    sprint_results = sprint_session.results
                    if sprint_results is not None and not sprint_results.empty:
                        # Map driver numbers to sprint points
                        for _, sprint_row in sprint_results.iterrows():
                            driver_num = sprint_row.get('DriverNumber')
                            sprint_points = sprint_row.get('Points', 0)
                            if pd.notna(sprint_points) and driver_num is not None:
                                sprint_points_dict[str(driver_num)] = sprint_points
                except Exception:
                    # No sprint race for this event, or sprint data not available
                    pass
                
                # Add sprint points to race points for total event points
                if sprint_points_dict:
                    results['SprintPoints'] = results['DriverNumber'].astype(str).map(sprint_points_dict).fillna(0)
                    # Total points = race points + sprint points
                    results['TotalEventPoints'] = results['Points'].fillna(0) + results['SprintPoints']
                else:
                    results['SprintPoints'] = 0
                    results['TotalEventPoints'] = results['Points'].fillna(0)
                
                # Try to get qualifying/starting grid position
                # Check if GridPosition column exists, if not try to get from qualifying session
                if 'GridPosition' not in results.columns:
                    try:
                        # Try to get qualifying session
                        qual_session = fastf1.get_session(year, event['EventName'], 'Q')
                        qual_session.load()
                        qual_results = qual_session.results
                        if qual_results is not None and not qual_results.empty:
                            # Merge qualifying positions
                            if 'Position' in qual_results.columns:
                                qual_map = dict(zip(qual_results['DriverNumber'], qual_results['Position']))
                                results['GridPosition'] = results['DriverNumber'].map(qual_map)
                    except Exception:
                        # If qualifying not available, try to use GridPosition from race results
                        # Some results have GridPosition directly
                        pass
                
                # If still no GridPosition, try to get from session data
                if 'GridPosition' not in results.columns or results['GridPosition'].isna().all():
                    try:
                        # Check if session has starting grid info
                        if hasattr(session, 'starting_grid') and session.starting_grid is not None:
                            grid_df = session.starting_grid
                            if 'Position' in grid_df.columns:
                                grid_map = dict(zip(grid_df['DriverNumber'], grid_df['Position']))
                                results['GridPosition'] = results['DriverNumber'].map(grid_map)
                    except Exception:
                        pass
                
                all_races.append(results)
        except Exception as e:
            print(f"  Warning: Error loading race {event.get('EventName', 'Unknown')} {year}: {e}")
            continue
    
    if all_races:
        return pd.concat(all_races, ignore_index=True)
    return pd.DataFrame()


def calculate_season_points(driver_results: pd.DataFrame) -> Dict[str, int]:
    """
    Calculate total points for each driver in a season.
    
    Args:
        driver_results: DataFrame with race results
        
    Returns:
        Dictionary mapping driver numbers to total season points
    """
    points_dict = {}
    for driver_num in driver_results['DriverNumber'].unique():
        driver_races = driver_results[driver_results['DriverNumber'] == driver_num]
        total_points = driver_races['Points'].sum()
        points_dict[str(driver_num)] = total_points
    return points_dict


def calculate_season_standing(driver_results: pd.DataFrame) -> Dict[str, int]:
    """
    Calculate championship position for each driver based on points.
    1 = leader (most points), higher = worse position.
    
    Args:
        driver_results: DataFrame with race results
        
    Returns:
        Dictionary mapping driver numbers to championship position (1-20)
    """
    # Calculate points per driver
    points_dict = calculate_season_points(driver_results)
    
    if not points_dict:
        return {}
    
    # Sort drivers by points (descending)
    sorted_drivers = sorted(points_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Assign positions (1 = most points)
    standing_dict = {}
    position = 1
    prev_points = None
    
    for driver_num, points in sorted_drivers:
        # If points are the same as previous driver, they share the position
        if prev_points is not None and points < prev_points:
            position = len(standing_dict) + 1
        standing_dict[str(driver_num)] = position
        prev_points = points
    
    return standing_dict


def calculate_season_avg_finish(driver_results: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate average finish position for each driver in a season.
    
    Args:
        driver_results: DataFrame with race results
        
    Returns:
        Dictionary mapping driver numbers to average finish position
    """
    avg_finish_dict = {}
    for driver_num in driver_results['DriverNumber'].unique():
        driver_races = driver_results[driver_results['DriverNumber'] == driver_num]
        # PositionText might have 'DNF', 'DSQ', etc., so we use Position
        valid_positions = driver_races['Position'].dropna()
        if len(valid_positions) > 0:
            avg_finish_dict[str(driver_num)] = valid_positions.mean()
        else:
            avg_finish_dict[str(driver_num)] = np.nan
    return avg_finish_dict


def calculate_constructor_points(driver_results: pd.DataFrame) -> Dict[str, int]:
    """
    Calculate total constructor points for each driver's team.
    
    Args:
        driver_results: DataFrame with race results
        
    Returns:
        Dictionary mapping driver numbers to their constructor's total points
    """
    constructor_points_dict = {}
    
    # Group by constructor and sum points
    if 'TeamName' in driver_results.columns:
        constructor_points = driver_results.groupby('TeamName')['Points'].sum().to_dict()
        
        # Map each driver to their constructor's points
        for driver_num in driver_results['DriverNumber'].unique():
            driver_races = driver_results[driver_results['DriverNumber'] == driver_num]
            if not driver_races.empty and 'TeamName' in driver_races.columns:
                team_name = driver_races['TeamName'].iloc[0]
                constructor_points_dict[str(driver_num)] = constructor_points.get(team_name, 0)
            else:
                constructor_points_dict[str(driver_num)] = 0
    else:
        # Fallback: if TeamName not available, use 0
        for driver_num in driver_results['DriverNumber'].unique():
            constructor_points_dict[str(driver_num)] = 0
    
    return constructor_points_dict


def calculate_constructor_standing(driver_results: pd.DataFrame) -> Dict[str, int]:
    """
    Calculate constructor championship standing (1 = best, higher = worse).
    
    Args:
        driver_results: DataFrame with race results
        
    Returns:
        Dictionary mapping driver numbers to their constructor's standing
    """
    constructor_standing_dict = {}
    
    if 'TeamName' in driver_results.columns:
        # Calculate constructor points
        constructor_points = driver_results.groupby('TeamName')['Points'].sum().sort_values(ascending=False)
        
        # Assign standings (1 = most points, higher = fewer points)
        constructor_standings = {team: rank + 1 for rank, team in enumerate(constructor_points.index)}
        
        # Map each driver to their constructor's standing
        for driver_num in driver_results['DriverNumber'].unique():
            driver_races = driver_results[driver_results['DriverNumber'] == driver_num]
            if not driver_races.empty and 'TeamName' in driver_races.columns:
                team_name = driver_races['TeamName'].iloc[0]
                constructor_standing_dict[str(driver_num)] = constructor_standings.get(team_name, 10)  # Default to 10 if unknown
            else:
                constructor_standing_dict[str(driver_num)] = 10
    else:
        # Fallback: if TeamName not available, use 10 (mid-field)
        for driver_num in driver_results['DriverNumber'].unique():
            constructor_standing_dict[str(driver_num)] = 10
    
    return constructor_standing_dict


def calculate_recent_grid_avg(driver_results: pd.DataFrame, num_races: int = 5) -> Dict[str, float]:
    """
    Calculate recent average grid position (qualifying performance).
    
    Args:
        driver_results: DataFrame with race results, sorted by RoundNumber
        num_races: Number of recent races to consider (default: 5)
        
    Returns:
        Dictionary mapping driver numbers to recent average grid position
    """
    recent_grid_dict = {}
    
    # Sort by round number to get most recent races
    driver_results = driver_results.sort_values('RoundNumber', ascending=False)
    
    for driver_num in driver_results['DriverNumber'].unique():
        driver_races = driver_results[driver_results['DriverNumber'] == driver_num]
        # Get last N races
        recent_races = driver_races.head(num_races)
        valid_grid = recent_races['GridPosition'].dropna()
        if len(valid_grid) > 0:
            recent_grid_dict[str(driver_num)] = valid_grid.mean()
        else:
            recent_grid_dict[str(driver_num)] = np.nan
    
    return recent_grid_dict


def calculate_constructor_recent_form(driver_results: pd.DataFrame, num_races: int = 5) -> Dict[str, float]:
    """
    Calculate constructor's recent form (average finish position of both drivers in last N races).
    
    Args:
        driver_results: DataFrame with race results, sorted by RoundNumber
        num_races: Number of recent races to consider (default: 5)
        
    Returns:
        Dictionary mapping constructor names to recent average finish position
    """
    constructor_form_dict = {}
    
    # Sort by round number to get most recent races
    driver_results = driver_results.sort_values('RoundNumber', ascending=False)
    
    # Group by constructor
    if 'TeamName' in driver_results.columns:
        for constructor in driver_results['TeamName'].unique():
            constructor_races = driver_results[driver_results['TeamName'] == constructor]
            # Get last N races (across both drivers)
            recent_races = constructor_races.head(num_races * 2)  # *2 because 2 drivers per team
            valid_positions = recent_races['Position'].dropna()
            if len(valid_positions) > 0:
                constructor_form_dict[constructor] = valid_positions.mean()
            else:
                constructor_form_dict[constructor] = np.nan
    else:
        # Fallback: use constructor from driver number mapping if available
        # For now, return empty dict
        pass
    
    return constructor_form_dict


def calculate_recent_form(driver_results: pd.DataFrame, num_races: int = 5) -> Dict[str, float]:
    """
    Calculate recent form (average finish position in last N races).
    
    Args:
        driver_results: DataFrame with race results, sorted by RoundNumber
        num_races: Number of recent races to consider (default: 5)
        
    Returns:
        Dictionary mapping driver numbers to recent average finish position
    """
    recent_form_dict = {}
    
    # Sort by round number to get most recent races
    driver_results = driver_results.sort_values('RoundNumber', ascending=False)
    
    for driver_num in driver_results['DriverNumber'].unique():
        driver_races = driver_results[driver_results['DriverNumber'] == driver_num]
        # Get last N races
        recent_races = driver_races.head(num_races)
        valid_positions = recent_races['Position'].dropna()
        if len(valid_positions) > 0:
            recent_form_dict[str(driver_num)] = valid_positions.mean()
        else:
            recent_form_dict[str(driver_num)] = np.nan
    
    return recent_form_dict


def calculate_track_avg_position(driver_results: pd.DataFrame, track_name: str) -> Dict[str, float]:
    """
    Calculate historical average position for each driver at a specific track.
    
    Args:
        driver_results: DataFrame with all historical race results
        track_name: Name of the track
        
    Returns:
        Dictionary mapping driver numbers to average position at this track
    """
    track_races = driver_results[driver_results['EventName'] == track_name]
    track_avg_dict = {}
    
    for driver_num in track_races['DriverNumber'].unique():
        driver_track_races = track_races[track_races['DriverNumber'] == driver_num]
        valid_positions = driver_track_races['Position'].dropna()
        if len(valid_positions) > 0:
            track_avg_dict[str(driver_num)] = valid_positions.mean()
        else:
            track_avg_dict[str(driver_num)] = 15.0  # Default for rookies
    
    return track_avg_dict


def calculate_constructor_track_avg(df: pd.DataFrame, constructor_standing: int, 
                                     track_name: str, current_year: int, current_round: int) -> float:
    """
    Calculate constructor's average finish at this specific track.
    Uses constructor standing as proxy for constructor identity.
    
    Args:
        df: DataFrame with all race data (may not have ConstructorStanding column)
        constructor_standing: Constructor's championship standing (1 = best, higher = worse)
        track_name: Name of the track
        current_year: Current race year
        current_round: Current race round number
        
    Returns:
        Average finish position for this constructor at this track (lower = better)
    """
    # Filter to races before current race (to avoid data leakage)
    historical_races = df[
        (df['EventName'] == track_name) &
        ((df['Year'] < current_year) | ((df['Year'] == current_year) & (df['RoundNumber'] < current_round)))
    ].copy()
    
    if historical_races.empty:
        return np.nan
    
    # If ConstructorStanding column exists, use it directly
    if 'ConstructorStanding' in historical_races.columns:
        track_races = historical_races[historical_races['ConstructorStanding'] == constructor_standing]
    else:
        # Calculate constructor standing on the fly using TeamName and Points
        if 'TeamName' not in historical_races.columns or 'Points' not in historical_races.columns:
            return np.nan
        
        # For each year, calculate constructor standings
        track_races = pd.DataFrame()
        for year in historical_races['Year'].unique():
            year_races = historical_races[historical_races['Year'] == year]
            if year_races.empty:
                continue
            
            # Calculate constructor points for this year
            constructor_points = year_races.groupby('TeamName')['Points'].sum().sort_values(ascending=False)
            constructor_standings = {team: rank + 1 for rank, team in enumerate(constructor_points.index)}
            
            # Filter to constructors with the target standing
            target_teams = [team for team, standing in constructor_standings.items() if standing == constructor_standing]
            if target_teams:
                year_track_races = year_races[year_races['TeamName'].isin(target_teams)]
                track_races = pd.concat([track_races, year_track_races], ignore_index=True)
    
    if track_races.empty:
        return np.nan
    
    # Use ActualPosition if available, otherwise Position
    pos_col = 'ActualPosition' if 'ActualPosition' in track_races.columns else 'Position'
    if pos_col in track_races.columns:
        positions = track_races[pos_col].dropna()
        if len(positions) > 0:
            return positions.mean()
    
    return np.nan


def is_street_circuit(track_name: str) -> int:
    """
    Determine if a track is a street circuit (1) or permanent circuit (0).
    Street circuits are more unpredictable.
    """
    street_circuits = [
        'Monaco', 'Singapore', 'Azerbaijan', 'Miami', 'Las Vegas',
        'Saudi Arabian'
    ]
    return 1 if any(street in track_name for street in street_circuits) else 0


def calculate_form_trend(driver_results: pd.DataFrame, driver_num: str, current_round: int) -> float:
    """
    Calculate form trend: difference between last 3 races avg and previous 3 races avg.
    Positive = improving (lower positions = better), Negative = declining.
    
    Args:
        driver_results: DataFrame with race results for a season, sorted by RoundNumber
        driver_num: Driver number as string
        current_round: Current round number (to avoid data leakage)
        
    Returns:
        Form trend value (positive = improving, negative = declining)
    """
    driver_races = driver_results[
        (driver_results['DriverNumber'] == driver_num) & 
        (driver_results['RoundNumber'] < current_round)
    ].copy()
    
    if len(driver_races) < 6:
        return 0.0  # Not enough data
    
    driver_races = driver_races.sort_values('RoundNumber', ascending=False)
    
    # Last 3 races
    last_3 = driver_races.head(3)
    last_3_avg = last_3['Position'].dropna().mean()
    
    # Previous 3 races (races 4-6)
    if len(driver_races) >= 6:
        prev_3 = driver_races.iloc[3:6]
        prev_3_avg = prev_3['Position'].dropna().mean()
    else:
        return 0.0
    
    if pd.isna(last_3_avg) or pd.isna(prev_3_avg):
        return 0.0
    
    # Negative means improving (lower position = better)
    # So: prev_avg - last_avg = positive if improving
    trend = prev_3_avg - last_3_avg
    return trend if not pd.isna(trend) else 0.0


def calculate_average_grid_position(df: pd.DataFrame, driver_num: str, current_year: int, current_round: int, season_specific: bool = True) -> float:
    """
    Calculate average grid position for a driver using only data up to the current race.
    This simulates what we'd use for future race predictions.
    
    Args:
        df: DataFrame with all race data
        driver_num: Driver number as string
        current_year: Current race year
        current_round: Current race round number
        season_specific: If True, only use races from current season. If False, use all-time history.
        
    Returns:
        Average grid position for the driver
    """
    if season_specific:
        # Only use races from the current season (before current round)
        driver_races = df[
            (df['DriverNumber'] == driver_num) &
            (df['Year'] == current_year) &
            (df['RoundNumber'] < current_round)
        ]
    else:
        # Use all-time history (all races before current race)
        driver_races = df[
            (df['DriverNumber'] == driver_num) &
            ((df['Year'] < current_year) | 
             ((df['Year'] == current_year) & (df['RoundNumber'] < current_round)))
        ]
    
    if driver_races.empty:
        return np.nan
    
    # Get grid positions
    grid_positions = driver_races['GridPosition'].dropna()
    
    if len(grid_positions) > 0:
        return grid_positions.mean()
    else:
        return np.nan


def organize_data(training_years: List[int], test_year: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Collect and organize F1 data into features and labels.
    
    Args:
        training_years: List of years for training data (e.g., [2020, 2021, 2022, 2023, 2024])
        test_year: Year for test data (e.g., 2025)
        
    Returns:
        Tuple of (training_data, test_data) DataFrames
    """
    print("Collecting training data...")
    training_races = []
    
    # Collect training data from past seasons
    successful_years = []
    for year in training_years:
        print(f"  Loading season {year}...")
        season_data = get_season_data(year)
        if not season_data.empty:
            training_races.append(season_data)
            successful_years.append(year)
            print(f"  Successfully loaded {len(season_data)} race results from {year}")
        else:
            print(f"  No data collected for {year}")
    
    if not training_races:
        raise ValueError(
            "No training data collected! This may be due to:\n"
            "  - Network connectivity issues\n"
            "  - Fast F1 API being temporarily unavailable\n"
            "  - Rate limiting from the API\n"
            "  - Invalid year ranges\n"
            "\nTry running again later, or check your internet connection."
        )
    
    print(f"\nSuccessfully collected data from {len(successful_years)}/{len(training_years)} training seasons: {successful_years}")
    
    all_training_data = pd.concat(training_races, ignore_index=True)
    
    # Collect test data
    print(f"Collecting test data for {test_year}...")
    test_data = get_season_data(test_year)
    
    # Organize features and labels
    print("Organizing features and labels...")
    
    # Dictionary to track historical sector times per (track, driver) for accumulation
    training_features = []
    test_features = []
    
    # Process training data
    for year in training_years:
        year_data = all_training_data[all_training_data['Year'] == year].copy()
        if year_data.empty:
            continue
        
        # Sort by round number to process races in order
        year_data = year_data.sort_values('RoundNumber')
        
        # For each race in the season, create feature vectors
        # IMPORTANT: Use only data available UP TO this race (no future data leakage)
        for idx, race in year_data.iterrows():
            driver_num = str(race['DriverNumber'])
            track_name = race['EventName']
            round_num = race['RoundNumber']
            
            # Calculate features using only races BEFORE this one (to avoid data leakage)
            # For first race of season, use previous season's data
            if round_num == 1:
                # Use previous year's data for first race
                prev_year_data = all_training_data[all_training_data['Year'] < year]
                if not prev_year_data.empty:
                    season_points = calculate_season_points(prev_year_data)
                    season_standing = calculate_season_standing(prev_year_data)
                    season_avg_finish = calculate_season_avg_finish(prev_year_data)
                    constructor_points = calculate_constructor_points(prev_year_data)
                    constructor_standing = calculate_constructor_standing(prev_year_data)
                    # Get points and avg from previous season
                    driver_points = season_points.get(driver_num, 0)
                    driver_standing = season_standing.get(driver_num, 20)  # Default to worst position if not found
                    driver_avg_finish = season_avg_finish.get(driver_num, np.nan)
                    driver_constructor_points = constructor_points.get(driver_num, 0)
                    driver_constructor_standing = constructor_standing.get(driver_num, 10)
                else:
                    driver_points = 0
                    driver_standing = 20  # No previous data - worst position
                    driver_avg_finish = np.nan
                    driver_constructor_points = 0
                    driver_constructor_standing = 10
            else:
                # Use races from current season up to (but not including) this race
                races_up_to_now = year_data[year_data['RoundNumber'] < round_num]
                if not races_up_to_now.empty:
                    season_points = calculate_season_points(races_up_to_now)
                    season_standing = calculate_season_standing(races_up_to_now)
                    season_avg_finish = calculate_season_avg_finish(races_up_to_now)
                    constructor_points = calculate_constructor_points(races_up_to_now)
                    constructor_standing = calculate_constructor_standing(races_up_to_now)
                    driver_points = season_points.get(driver_num, 0)
                    driver_standing = season_standing.get(driver_num, 20)  # Default to worst position if not found
                    driver_avg_finish = season_avg_finish.get(driver_num, np.nan)
                    driver_constructor_points = constructor_points.get(driver_num, 0)
                    driver_constructor_standing = constructor_standing.get(driver_num, 10)
                else:
                    driver_points = 0
                    driver_standing = 20  # No races yet - worst position
                    driver_avg_finish = np.nan
                    driver_constructor_points = 0
                    driver_constructor_standing = 10
            
            # Historical track average (excluding current race and future races)
            historical_data = all_training_data[
                (all_training_data['EventName'] == track_name) & 
                ((all_training_data['Year'] < year) | 
                 ((all_training_data['Year'] == year) & (all_training_data['RoundNumber'] < round_num)))
            ]
            track_avg = calculate_track_avg_position(historical_data, track_name)
            
            # Get historical track average for this driver, or use fallback
            hist_track_avg = track_avg.get(driver_num, np.nan)
            
            # Fallback: If no track-specific history, use driver's overall average position
            if pd.isna(hist_track_avg):
                driver_all_races = all_training_data[
                    (all_training_data['DriverNumber'] == race['DriverNumber']) &
                    ((all_training_data['Year'] < year) | 
                     ((all_training_data['Year'] == year) & (all_training_data['RoundNumber'] < round_num)))
                ]
                if not driver_all_races.empty:
                    valid_positions = driver_all_races['Position'].dropna()
                    if len(valid_positions) > 0:
                        hist_track_avg = valid_positions.mean()  # Overall career average
            
            # Get starting grid position - use AVERAGE grid position instead of actual
            # This matches what we'll use for future race predictions and eliminates train/test mismatch
            # Using season-specific average (only races from current season before current round)
            grid_position = calculate_average_grid_position(all_training_data, driver_num, year, round_num, season_specific=True)
            
            # Fallback: if no historical data, use actual grid position from this race
            if pd.isna(grid_position):
                grid_position = race.get('GridPosition', np.nan)
                if pd.isna(grid_position):
                    # Try alternative column names
                    grid_position = race.get('StartingGrid', race.get('Grid', np.nan))
            
            # Recent form (last 5 races average finish) - captures current momentum
            if round_num == 1:
                # First race: use previous season's recent form
                prev_year_data = all_training_data[all_training_data['Year'] < year]
                if not prev_year_data.empty:
                    recent_form = calculate_recent_form(prev_year_data, num_races=5)
                    driver_recent_form = recent_form.get(driver_num, np.nan)
                else:
                    driver_recent_form = np.nan
            else:
                # Use races from current season up to (but not including) this race
                races_up_to_now = year_data[year_data['RoundNumber'] < round_num]
                if not races_up_to_now.empty:
                    recent_form = calculate_recent_form(races_up_to_now, num_races=5)
                    driver_recent_form = recent_form.get(driver_num, np.nan)
                else:
                    driver_recent_form = np.nan
            
            # Calculate new features for improved model
            # 1. PointsGapToLeader: Points gap to championship leader
            if round_num == 1:
                prev_year_data = all_training_data[all_training_data['Year'] < year]
                if not prev_year_data.empty:
                    prev_season_points = calculate_season_points(prev_year_data)
                    max_points = max(prev_season_points.values()) if prev_season_points else 0
                    points_gap = max_points - driver_points if max_points > 0 else 0
                else:
                    points_gap = 0
            else:
                races_up_to_now = year_data[year_data['RoundNumber'] < round_num]
                if not races_up_to_now.empty:
                    season_points = calculate_season_points(races_up_to_now)
                    max_points = max(season_points.values()) if season_points else 0
                    points_gap = max_points - driver_points if max_points > 0 else 0
                else:
                    points_gap = 0
            
            # 2. TrackType: 1 for street circuit, 0 for permanent
            track_type = is_street_circuit(track_name)
            
            # 3. ConstructorTrackAvg: Constructor's average finish at this specific track
            constructor_track_avg = calculate_constructor_track_avg(
                all_training_data, driver_constructor_standing, track_name, year, round_num
            )
            # Fallback: if no constructor track history, use constructor's overall average
            if pd.isna(constructor_track_avg):
                # Calculate constructor's overall average using TeamName if available
                historical_races = all_training_data[
                    ((all_training_data['Year'] < year) | 
                     ((all_training_data['Year'] == year) & (all_training_data['RoundNumber'] < round_num)))
                ].copy()
                
                if not historical_races.empty and 'TeamName' in historical_races.columns and 'Points' in historical_races.columns:
                    # Find teams with the target constructor standing
                    # Calculate standings for the most recent year available
                    latest_year = historical_races['Year'].max()
                    year_data = historical_races[historical_races['Year'] == latest_year]
                    if not year_data.empty:
                        constructor_points = year_data.groupby('TeamName')['Points'].sum().sort_values(ascending=False)
                        constructor_standings = {team: rank + 1 for rank, team in enumerate(constructor_points.index)}
                        target_teams = [team for team, standing in constructor_standings.items() if standing == driver_constructor_standing]
                        
                        if target_teams:
                            constructor_all_races = historical_races[historical_races['TeamName'].isin(target_teams)]
                            pos_col = 'ActualPosition' if 'ActualPosition' in constructor_all_races.columns else 'Position'
                            if pos_col in constructor_all_races.columns:
                                valid_positions = constructor_all_races[pos_col].dropna()
                                if len(valid_positions) > 0:
                                    constructor_track_avg = valid_positions.mean()
                                else:
                                    constructor_track_avg = 10.0
                            else:
                                constructor_track_avg = 10.0
                        else:
                            constructor_track_avg = 10.0
                    else:
                        constructor_track_avg = 10.0
                else:
                    constructor_track_avg = 10.0  # Default mid-field
            
            # 4. FormTrend: Momentum direction (improving vs declining)
            if round_num == 1:
                # First race: use previous season's trend
                prev_year_data = all_training_data[all_training_data['Year'] < year]
                if not prev_year_data.empty:
                    # Get last round of previous season
                    prev_year_sorted = prev_year_data.sort_values('RoundNumber')
                    if not prev_year_sorted.empty:
                        last_round = prev_year_sorted['RoundNumber'].max()
                        form_trend = calculate_form_trend(prev_year_sorted, driver_num, last_round + 1)
                    else:
                        form_trend = 0.0
                else:
                    form_trend = 0.0
            else:
                # Use current season data
                form_trend = calculate_form_trend(year_data, driver_num, round_num)
            
            # Get DNF status if available
            status = race.get('Status', '')
            position_text = race.get('PositionText', '')
            is_dnf = False
            if pd.notna(status):
                status_str = str(status).upper()
                is_dnf = any(x in status_str for x in ['DNF', 'DSQ', 'DNS', 'NC', 'DISQUALIFIED', 'NOT CLASSIFIED'])
            elif pd.notna(position_text):
                pos_text_str = str(position_text).upper()
                is_dnf = any(x in pos_text_str for x in ['DNF', 'DSQ', 'DNS', 'NC'])
            
            features = {
                'Year': year,
                'EventName': track_name,
                'RoundNumber': round_num,
                'SeasonPoints': driver_points,  # Keep for backward compatibility
                'SeasonStanding': driver_standing,  # Championship position (1 = leader, higher = worse)
                'SeasonAvgFinish': driver_avg_finish,
                'HistoricalTrackAvgPosition': hist_track_avg,
                'ConstructorPoints': driver_constructor_points,
                'ConstructorStanding': driver_constructor_standing,
                'ConstructorTrackAvg': constructor_track_avg,  # Constructor's average finish at this track
                'GridPosition': grid_position,  # Average grid position (matches future prediction scenario)
                'RecentForm': driver_recent_form,  # Last 5 races average finish (current momentum)
                'PointsGapToLeader': points_gap,  # Points gap to championship leader
                'TrackType': track_type,  # 1 = street circuit, 0 = permanent
                'FormTrend': form_trend,  # Momentum direction (positive = improving)
                'DriverNumber': race['DriverNumber'],
                'DriverName': race.get('Abbreviation', 'UNK'),
                'TeamName': race.get('TeamName', race.get('Team', '')),  # Constructor/team name
                'ActualPosition': race.get('Position', np.nan),
                'Points': race.get('TotalEventPoints', race.get('Points', 0)),  # Race + Sprint points (source of truth)
                'RacePoints': race.get('Points', 0),  # Race points only
                'SprintPoints': race.get('SprintPoints', 0),  # Sprint points only
                'IsDNF': is_dnf,  # Flag for DNF/DSQ/DNS
                'Status': status if pd.notna(status) else position_text if pd.notna(position_text) else ''
            }
            training_features.append(features)
    
    # Process test data similarly (no data leakage - use only data up to each race)
    if not test_data.empty:
        test_data_sorted = test_data.sort_values('RoundNumber')
        
        for idx, race in test_data_sorted.iterrows():
            driver_num = str(race['DriverNumber'])
            track_name = race['EventName']
            round_num = race['RoundNumber']
            
            # Calculate features using only races BEFORE this one
            if round_num == 1:
                # Use last year's data for first race of test season
                last_year_data = all_training_data[all_training_data['Year'] == max(training_years)]
                if not last_year_data.empty:
                    season_points = calculate_season_points(last_year_data)
                    season_standing = calculate_season_standing(last_year_data)
                    season_avg_finish = calculate_season_avg_finish(last_year_data)
                    constructor_points = calculate_constructor_points(last_year_data)
                    constructor_standing = calculate_constructor_standing(last_year_data)
                    driver_points = season_points.get(driver_num, 0)
                    driver_standing = season_standing.get(driver_num, 20)  # Default to worst position if not found
                    driver_avg_finish = season_avg_finish.get(driver_num, np.nan)
                    driver_constructor_points = constructor_points.get(driver_num, 0)
                    driver_constructor_standing = constructor_standing.get(driver_num, 10)
                else:
                    driver_points = 0
                    driver_standing = 20  # No previous data - worst position
                    driver_avg_finish = np.nan
                    driver_constructor_points = 0
                    driver_constructor_standing = 10
            else:
                # Use races from test season up to (but not including) this race
                races_up_to_now = test_data_sorted[test_data_sorted['RoundNumber'] < round_num]
                if not races_up_to_now.empty:
                    season_points = calculate_season_points(races_up_to_now)
                    season_standing = calculate_season_standing(races_up_to_now)
                    season_avg_finish = calculate_season_avg_finish(races_up_to_now)
                    constructor_points = calculate_constructor_points(races_up_to_now)
                    constructor_standing = calculate_constructor_standing(races_up_to_now)
                    driver_points = season_points.get(driver_num, 0)
                    driver_standing = season_standing.get(driver_num, 20)  # Default to worst position if not found
                    driver_avg_finish = season_avg_finish.get(driver_num, np.nan)
                    driver_constructor_points = constructor_points.get(driver_num, 0)
                    driver_constructor_standing = constructor_standing.get(driver_num, 10)
                else:
                    driver_points = 0
                    driver_standing = 20  # No races yet - worst position
                    driver_avg_finish = np.nan
                    driver_constructor_points = 0
                    driver_constructor_standing = 10
            
            # Historical track average from training data only
            historical_data = all_training_data[all_training_data['EventName'] == track_name]
            track_avg = calculate_track_avg_position(historical_data, track_name)
            
            # Get historical track average for this driver, or use fallback
            hist_track_avg = track_avg.get(driver_num, np.nan)
            
            # Fallback: If no track-specific history, use driver's overall average from training data
            if pd.isna(hist_track_avg):
                driver_all_races = all_training_data[all_training_data['DriverNumber'] == race['DriverNumber']]
                if not driver_all_races.empty:
                    valid_positions = driver_all_races['Position'].dropna()
                    if len(valid_positions) > 0:
                        hist_track_avg = valid_positions.mean()  # Overall career average from training data
                    else:
                        hist_track_avg = 15.0  # Default for rookies with no history
                else:
                    hist_track_avg = 15.0  # Default for rookies with no history
            
            # Get starting grid position - use AVERAGE grid position instead of actual
            # This matches what we'll use for future race predictions and eliminates train/test mismatch
            # For test data, we calculate average from training data + previous test races
            if round_num == 1:
                # First race: use only training data
                combined_data = all_training_data
            else:
                # Use training data + previous test races
                previous_test_races = test_data_sorted[test_data_sorted['RoundNumber'] < round_num]
                combined_data = pd.concat([all_training_data, previous_test_races], ignore_index=True) if not previous_test_races.empty else all_training_data
            
            # Using season-specific average (only races from current season before current round)
            grid_position = calculate_average_grid_position(combined_data, driver_num, test_year, round_num, season_specific=True)
            
            # Fallback: if no historical data, use actual grid position from this race
            if pd.isna(grid_position):
                grid_position = race.get('GridPosition', np.nan)
                if pd.isna(grid_position):
                    # Try alternative column names
                    grid_position = race.get('StartingGrid', race.get('Grid', np.nan))
            
            # Recent form (last 5 races average finish) - captures current momentum
            if round_num == 1:
                # First race: use last year's recent form
                last_year_data = all_training_data[all_training_data['Year'] == max(training_years)]
                if not last_year_data.empty:
                    recent_form = calculate_recent_form(last_year_data, num_races=5)
                    driver_recent_form = recent_form.get(driver_num, np.nan)
                else:
                    driver_recent_form = np.nan
            else:
                # Use races from test season up to (but not including) this race
                races_up_to_now = test_data_sorted[test_data_sorted['RoundNumber'] < round_num]
                if not races_up_to_now.empty:
                    recent_form = calculate_recent_form(races_up_to_now, num_races=5)
                    driver_recent_form = recent_form.get(driver_num, np.nan)
                else:
                    driver_recent_form = np.nan
            
            # Calculate new features for improved model
            # 1. PointsGapToLeader: Points gap to championship leader
            if round_num == 1:
                last_year_data = all_training_data[all_training_data['Year'] == max(training_years)]
                if not last_year_data.empty:
                    last_season_points = calculate_season_points(last_year_data)
                    max_points = max(last_season_points.values()) if last_season_points else 0
                    points_gap = max_points - driver_points if max_points > 0 else 0
                else:
                    points_gap = 0
            else:
                races_up_to_now = test_data_sorted[test_data_sorted['RoundNumber'] < round_num]
                if not races_up_to_now.empty:
                    season_points = calculate_season_points(races_up_to_now)
                    max_points = max(season_points.values()) if season_points else 0
                    points_gap = max_points - driver_points if max_points > 0 else 0
                else:
                    points_gap = 0
            
            # 2. TrackType: 1 for street circuit, 0 for permanent
            track_type = is_street_circuit(track_name)
            
            # 3. ConstructorTrackAvg: Constructor's average finish at this specific track
            # Use combined data (training + previous test races) for calculation
            constructor_track_avg = calculate_constructor_track_avg(
                combined_data, driver_constructor_standing, track_name, test_year, round_num
            )
            # Fallback: if no constructor track history, use constructor's overall average
            if pd.isna(constructor_track_avg):
                # Calculate constructor's overall average using TeamName if available
                historical_races = combined_data[
                    ((combined_data['Year'] < test_year) | 
                     ((combined_data['Year'] == test_year) & (combined_data['RoundNumber'] < round_num)))
                ].copy()
                
                if not historical_races.empty and 'TeamName' in historical_races.columns and 'Points' in historical_races.columns:
                    # Find teams with the target constructor standing
                    # Calculate standings for the most recent year available
                    latest_year = historical_races['Year'].max()
                    year_data = historical_races[historical_races['Year'] == latest_year]
                    if not year_data.empty:
                        constructor_points = year_data.groupby('TeamName')['Points'].sum().sort_values(ascending=False)
                        constructor_standings = {team: rank + 1 for rank, team in enumerate(constructor_points.index)}
                        target_teams = [team for team, standing in constructor_standings.items() if standing == driver_constructor_standing]
                        
                        if target_teams:
                            constructor_all_races = historical_races[historical_races['TeamName'].isin(target_teams)]
                            pos_col = 'ActualPosition' if 'ActualPosition' in constructor_all_races.columns else 'Position'
                            if pos_col in constructor_all_races.columns:
                                valid_positions = constructor_all_races[pos_col].dropna()
                                if len(valid_positions) > 0:
                                    constructor_track_avg = valid_positions.mean()
                                else:
                                    constructor_track_avg = 10.0
                            else:
                                constructor_track_avg = 10.0
                        else:
                            constructor_track_avg = 10.0
                    else:
                        constructor_track_avg = 10.0
                else:
                    constructor_track_avg = 10.0  # Default mid-field
            
            # 4. FormTrend: Momentum direction (improving vs declining)
            if round_num == 1:
                # First race: use last year's trend
                last_year_data = all_training_data[all_training_data['Year'] == max(training_years)]
                if not last_year_data.empty:
                    last_year_sorted = last_year_data.sort_values('RoundNumber')
                    if not last_year_sorted.empty:
                        last_round = last_year_sorted['RoundNumber'].max()
                        form_trend = calculate_form_trend(last_year_sorted, driver_num, last_round + 1)
                    else:
                        form_trend = 0.0
                else:
                    form_trend = 0.0
            else:
                # Use test season data
                form_trend = calculate_form_trend(test_data_sorted, driver_num, round_num)
            
            # Get DNF status if available
            status = race.get('Status', '')
            position_text = race.get('PositionText', '')
            is_dnf = False
            if pd.notna(status):
                status_str = str(status).upper()
                is_dnf = any(x in status_str for x in ['DNF', 'DSQ', 'DNS', 'NC', 'DISQUALIFIED', 'NOT CLASSIFIED'])
            elif pd.notna(position_text):
                pos_text_str = str(position_text).upper()
                is_dnf = any(x in pos_text_str for x in ['DNF', 'DSQ', 'DNS', 'NC'])
            
            features = {
                'Year': test_year,
                'EventName': track_name,
                'RoundNumber': round_num,
                'SeasonPoints': driver_points,  # Keep for backward compatibility
                'SeasonStanding': driver_standing,  # Championship position (1 = leader, higher = worse)
                'SeasonAvgFinish': driver_avg_finish,
                'HistoricalTrackAvgPosition': hist_track_avg,
                'ConstructorPoints': driver_constructor_points,
                'ConstructorStanding': driver_constructor_standing,
                'ConstructorTrackAvg': constructor_track_avg,  # Constructor's average finish at this track
                'GridPosition': grid_position,  # Average grid position (matches future prediction scenario)
                'RecentForm': driver_recent_form,  # Last 5 races average finish (current momentum)
                'PointsGapToLeader': points_gap,  # Points gap to championship leader
                'TrackType': track_type,  # 1 = street circuit, 0 = permanent
                'FormTrend': form_trend,  # Momentum direction (positive = improving)
                'DriverNumber': race['DriverNumber'],
                'DriverName': race.get('Abbreviation', 'UNK'),
                'TeamName': race.get('TeamName', race.get('Team', '')),  # Constructor/team name
                'ActualPosition': race.get('Position', np.nan),
                'Points': race.get('TotalEventPoints', race.get('Points', 0)),  # Race + Sprint points (source of truth)
                'RacePoints': race.get('Points', 0),  # Race points only
                'SprintPoints': race.get('SprintPoints', 0),  # Sprint points only
                'IsDNF': is_dnf,  # Flag for DNF/DSQ/DNS
                'Status': status if pd.notna(status) else position_text if pd.notna(position_text) else ''
            }
            test_features.append(features)
    
    training_df = pd.DataFrame(training_features)
    test_df = pd.DataFrame(test_features)
    
    return training_df, test_df


def save_data(training_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str = 'data'):
    """
    Save organized data to CSV files.
    
    Args:
        training_df: Training data DataFrame
        test_df: Test data DataFrame
        output_dir: Directory to save data files
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    training_path = Path(output_dir) / 'training_data.csv'
    test_path = Path(output_dir) / 'test_data.csv'
    
    training_df.to_csv(training_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\nData saved:")
    print(f"  Training data: {training_path} ({len(training_df)} rows)")
    print(f"  Test data: {test_path} ({len(test_df)} rows)")
    
    # Save metadata
    metadata = {
        'training_samples': len(training_df),
        'test_samples': len(test_df),
        'features': ['SeasonPoints', 'SeasonAvgFinish', 'HistoricalTrackAvgPosition', 
                     'ConstructorPoints', 'ConstructorStanding', 'GridPosition', 'RecentForm', 
                     'TrackType'],
        'label': 'DriverNumber'
    }
    
    with open(Path(output_dir) / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)


def main():
    """Main function to collect and organize F1 data."""
    # Training data from 2022 onwards (more recent, relevant data)
    training_years = [2022, 2023, 2024]
    # Current season for testing (2025)
    test_year = 2025
    
    print("F1 Data Collection")
    print("=" * 50)
    print("Note: Data is cached locally. First run will download data,")
    print("      subsequent runs will use cached data (much faster).")
    print()
    print("Note: If you encounter API errors, the script will:")
    print("      - Retry failed requests up to 3 times")
    print("      - Skip problematic years and continue with available data")
    print("      - Continue even if some years fail to load")
    print()
    
    try:
        training_df, test_df = organize_data(training_years, test_year)
        save_data(training_df, test_df)
        
        print("\nData collection complete!")
        print(f"\nTraining data summary:")
        feature_cols = ['SeasonPoints', 'SeasonAvgFinish', 'HistoricalTrackAvgPosition', 
                       'ConstructorPoints', 'ConstructorStanding', 'GridPosition', 'RecentForm']
        print(training_df[feature_cols + ['DriverNumber']].describe())
        
    except Exception as e:
        print(f"Error during data collection: {e}")
        raise


if __name__ == "__main__":
    main()

