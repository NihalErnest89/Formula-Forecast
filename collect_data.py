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
            track_avg_dict[str(driver_num)] = np.nan
    
    return track_avg_dict


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
                    season_avg_finish = calculate_season_avg_finish(prev_year_data)
                    constructor_points = calculate_constructor_points(prev_year_data)
                    constructor_standing = calculate_constructor_standing(prev_year_data)
                    # Get points and avg from previous season
                    driver_points = season_points.get(driver_num, 0)
                    driver_avg_finish = season_avg_finish.get(driver_num, np.nan)
                    driver_constructor_points = constructor_points.get(driver_num, 0)
                    driver_constructor_standing = constructor_standing.get(driver_num, 10)
                else:
                    driver_points = 0
                    driver_avg_finish = np.nan
                    driver_constructor_points = 0
                    driver_constructor_standing = 10
            else:
                # Use races from current season up to (but not including) this race
                races_up_to_now = year_data[year_data['RoundNumber'] < round_num]
                if not races_up_to_now.empty:
                    season_points = calculate_season_points(races_up_to_now)
                    season_avg_finish = calculate_season_avg_finish(races_up_to_now)
                    constructor_points = calculate_constructor_points(races_up_to_now)
                    constructor_standing = calculate_constructor_standing(races_up_to_now)
                    driver_points = season_points.get(driver_num, 0)
                    driver_avg_finish = season_avg_finish.get(driver_num, np.nan)
                    driver_constructor_points = constructor_points.get(driver_num, 0)
                    driver_constructor_standing = constructor_standing.get(driver_num, 10)
                else:
                    driver_points = 0
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
            
            # Get starting grid position (qualifying position)
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
            
            features = {
                'Year': year,
                'EventName': track_name,
                'RoundNumber': round_num,
                'SeasonPoints': driver_points,
                'SeasonAvgFinish': driver_avg_finish,
                'HistoricalTrackAvgPosition': hist_track_avg,
                'ConstructorPoints': driver_constructor_points,
                'ConstructorStanding': driver_constructor_standing,
                'GridPosition': grid_position,  # Starting grid position (qualifying)
                'RecentForm': driver_recent_form,  # Last 5 races average finish (current momentum)
                'DriverNumber': race['DriverNumber'],
                'DriverName': race.get('Abbreviation', 'UNK'),
                'ActualPosition': race.get('Position', np.nan)
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
                    season_avg_finish = calculate_season_avg_finish(last_year_data)
                    constructor_points = calculate_constructor_points(last_year_data)
                    constructor_standing = calculate_constructor_standing(last_year_data)
                    driver_points = season_points.get(driver_num, 0)
                    driver_avg_finish = season_avg_finish.get(driver_num, np.nan)
                    driver_constructor_points = constructor_points.get(driver_num, 0)
                    driver_constructor_standing = constructor_standing.get(driver_num, 10)
                else:
                    driver_points = 0
                    driver_avg_finish = np.nan
                    driver_constructor_points = 0
                    driver_constructor_standing = 10
            else:
                # Use races from test season up to (but not including) this race
                races_up_to_now = test_data_sorted[test_data_sorted['RoundNumber'] < round_num]
                if not races_up_to_now.empty:
                    season_points = calculate_season_points(races_up_to_now)
                    season_avg_finish = calculate_season_avg_finish(races_up_to_now)
                    constructor_points = calculate_constructor_points(races_up_to_now)
                    constructor_standing = calculate_constructor_standing(races_up_to_now)
                    driver_points = season_points.get(driver_num, 0)
                    driver_avg_finish = season_avg_finish.get(driver_num, np.nan)
                    driver_constructor_points = constructor_points.get(driver_num, 0)
                    driver_constructor_standing = constructor_standing.get(driver_num, 10)
                else:
                    driver_points = 0
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
            
            # Get starting grid position (qualifying position)
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
            
            features = {
                'Year': test_year,
                'EventName': track_name,
                'RoundNumber': round_num,
                'SeasonPoints': driver_points,
                'SeasonAvgFinish': driver_avg_finish,
                'HistoricalTrackAvgPosition': hist_track_avg,
                'ConstructorPoints': driver_constructor_points,
                'ConstructorStanding': driver_constructor_standing,
                'GridPosition': grid_position,  # Starting grid position (qualifying)
                'RecentForm': driver_recent_form,  # Last 5 races average finish (current momentum)
                'DriverNumber': race['DriverNumber'],
                'DriverName': race.get('Abbreviation', 'UNK'),
                'ActualPosition': race.get('Position', np.nan)
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
                     'ConstructorPoints', 'ConstructorStanding', 'GridPosition', 'RecentForm'],
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

