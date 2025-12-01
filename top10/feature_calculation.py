import numpy as np
import pandas as pd
from pathlib import Path

def calculate_future_race_features(test_df: pd.DataFrame, selected_year: int, selected_round: int, 
                                    track_name: str, training_df: pd.DataFrame = None):
    """
    Calculate features for a future race using data from all completed races up to this point.
    
    Args:
        test_df: DataFrame with test data (completed races)
        selected_year: Year of the future race
        selected_round: Round number of the future race
        track_name: Name of the track for the future race
        
    Returns:
        DataFrame with driver features for the future race
    """
    # Get all completed races up to (but not including) the future race
    completed_races = test_df[
        (test_df['Year'] == selected_year) & 
        (test_df['RoundNumber'] < selected_round)
    ].copy()
    
    if completed_races.empty:
        # If no races in this year yet, use last year's final data
        last_year_races = test_df[test_df['Year'] < selected_year]
        if not last_year_races.empty:
            last_year = last_year_races['Year'].max()
            completed_races = test_df[test_df['Year'] == last_year].copy()
    
    if completed_races.empty:
        raise ValueError("No historical data available to calculate features for future race")
    
    # Get the most recent completed race for driver list and initial state
    most_recent_round = completed_races['RoundNumber'].max()
    most_recent_race = completed_races[completed_races['RoundNumber'] == most_recent_round].copy()
    
    print(f"  Calculating features from {len(completed_races)} completed races (most recent: Round {most_recent_round})")
    
    # For each driver in the most recent race, use their current features
    # These features represent their state up to the most recent race
    future_race_features = []
    
    # Calculate track-specific historical averages for this specific track
    # Use training data if available (has more historical data), otherwise use test data
    historical_df = training_df if training_df is not None and not training_df.empty else test_df

    track_historical_data = historical_df[historical_df['EventName'] == track_name]
    track_avg_by_driver = {}
    grid_avg_by_driver = {}  # Average grid position per driver
    
    for driver_num in track_historical_data['DriverNumber'].unique():
        driver_track_races = track_historical_data[track_historical_data['DriverNumber'] == driver_num]
        # Try to get ActualPosition or Position column
        if 'ActualPosition' in driver_track_races.columns:
            valid_positions = driver_track_races['ActualPosition'].dropna()
        elif 'Position' in driver_track_races.columns:
            valid_positions = driver_track_races['Position'].dropna()
        else:
            valid_positions = pd.Series()
        
        if len(valid_positions) > 0:
            track_avg_by_driver[str(driver_num)] = valid_positions.mean()
        else:
            # Fallback: use driver's overall average from historical data
            driver_all_races = historical_df[historical_df['DriverNumber'] == driver_num]
            if not driver_all_races.empty:
                if 'ActualPosition' in driver_all_races.columns:
                    valid_positions = driver_all_races['ActualPosition'].dropna()
                elif 'Position' in driver_all_races.columns:
                    valid_positions = driver_all_races['Position'].dropna()
                else:
                    valid_positions = pd.Series()
                
                if len(valid_positions) > 0:
                    track_avg_by_driver[str(driver_num)] = valid_positions.mean()
                else:
                    track_avg_by_driver[str(driver_num)] = 10.0  # Default for rookies
            else:
                track_avg_by_driver[str(driver_num)] = 10.0  # Default for rookies
        
        # Calculate average grid position for this driver (season-specific: only from current season)
        # Filter to only current season races before current round
        driver_season_races = historical_df[
            (historical_df['DriverNumber'] == driver_num) &
            (historical_df['Year'] == selected_year) &
            (historical_df['RoundNumber'] < selected_round)
        ]
        if not driver_season_races.empty and 'GridPosition' in driver_season_races.columns:
            valid_grid_positions = driver_season_races['GridPosition'].dropna()
            if len(valid_grid_positions) > 0:
                grid_avg_by_driver[str(driver_num)] = valid_grid_positions.mean()
            else:
                grid_avg_by_driver[str(driver_num)] = np.nan
        else:
            grid_avg_by_driver[str(driver_num)] = np.nan
    
    for _, driver_row in most_recent_race.iterrows():
        driver_num = driver_row['DriverNumber']
        driver_name = driver_row.get('DriverName', f"Driver {driver_num}")
        
        # Calculate cumulative features from ALL completed races (not just most recent)
        driver_completed_races = completed_races[completed_races['DriverNumber'] == driver_num].copy()
        
        # Calculate SeasonPoints: sum of all points from completed races
        # RELY ONLY ON API Points COLUMN - no manual calculation
        if not driver_completed_races.empty:
            if 'Points' in driver_completed_races.columns:
                # Use Points column directly from API (source of truth)
                season_points = driver_completed_races['Points'].sum()
            else:
                # Points column missing - use most recent race's cumulative points as fallback
                # This should rarely happen if data collection is correct
                season_points = driver_row.get('SeasonPoints', 0)
                if season_points == 0:
                    print(f"  Warning: No Points column found for driver {driver_num}, using 0 points")
        else:
            # No completed races - use most recent race's cumulative points
            season_points = driver_row.get('SeasonPoints', 0)
        
        # Calculate SeasonStanding: championship position based on points (1 = leader, higher = worse)
        # Calculate standings from all completed races in the current season
        if not driver_completed_races.empty:
            # Get all races from current season up to this point
            season_races = completed_races[completed_races['Year'] == selected_year].copy()
            if season_races.empty:
                # If no races in current season, use most recent year
                latest_year = completed_races['Year'].max()
                season_races = completed_races[completed_races['Year'] == latest_year].copy()
            
            if not season_races.empty and 'Points' in season_races.columns:
                # Calculate points per driver
                driver_points_dict = season_races.groupby('DriverNumber')['Points'].sum().sort_values(ascending=False)
                
                # Find this driver's position (1 = most points)
                if driver_num in driver_points_dict.index:
                    driver_total_points = driver_points_dict[driver_num]
                    # Count how many drivers have more points
                    position = (driver_points_dict > driver_total_points).sum() + 1
                    season_standing = position
                else:
                    # Driver has no points yet - assign worst position
                    season_standing = len(driver_points_dict) + 1 if len(driver_points_dict) > 0 else 20
            else:
                season_standing = driver_row.get('SeasonStanding', 20)  # Default to worst position
        else:
            # No completed races - use most recent race's standing or default
            season_standing = driver_row.get('SeasonStanding', 20)
        
        # Calculate SeasonAvgFinish: average of all completed race positions
        if not driver_completed_races.empty and 'ActualPosition' in driver_completed_races.columns:
            valid_positions = driver_completed_races['ActualPosition'].dropna()
            if len(valid_positions) > 0:
                season_avg_finish = valid_positions.mean()
            else:
                season_avg_finish = driver_row.get('SeasonAvgFinish', np.nan)
        else:
            season_avg_finish = driver_row.get('SeasonAvgFinish', np.nan)
        
        # Calculate RecentForm: average of last 5 completed races
        if not driver_completed_races.empty and 'ActualPosition' in driver_completed_races.columns:
            # Sort by round number descending to get most recent
            driver_races_sorted = driver_completed_races.sort_values('RoundNumber', ascending=False)
            last_5_races = driver_races_sorted.head(5)
            valid_positions = last_5_races['ActualPosition'].dropna()
            if len(valid_positions) > 0:
                recent_form = valid_positions.mean()
            else:
                recent_form = driver_row.get('RecentForm', np.nan)
        else:
            recent_form = driver_row.get('RecentForm', np.nan)
        
        # Calculate ConstructorPoints and ConstructorStanding from completed races
        # ALWAYS calculate dynamically from completed races - don't rely on stale values
        # First, get the driver's team name from most recent race or historical data
        team_name = None
        if 'TeamName' in driver_row and pd.notna(driver_row.get('TeamName')) and str(driver_row.get('TeamName')).strip():
            team_name = str(driver_row['TeamName']).strip()
        elif not driver_completed_races.empty and 'TeamName' in driver_completed_races.columns:
            # Try to get from completed races - use MOST RECENT race (by RoundNumber)
            # This handles cases where drivers switch teams mid-season
            driver_races_sorted = driver_completed_races.sort_values('RoundNumber', ascending=False)
            valid_team_names = driver_races_sorted['TeamName'].dropna()
            if not valid_team_names.empty:
                # Get team from most recent race
                team_name = str(valid_team_names.iloc[0]).strip() if str(valid_team_names.iloc[0]).strip() else None
        elif historical_df is not None and not historical_df.empty:
            # Try to get from historical data - use MOST RECENT race (by RoundNumber, then Year)
            driver_historical = historical_df[historical_df['DriverNumber'] == driver_num]
            if not driver_historical.empty and 'TeamName' in driver_historical.columns:
                # Sort by Year and RoundNumber descending to get most recent race first
                if 'Year' in driver_historical.columns and 'RoundNumber' in driver_historical.columns:
                    driver_historical_sorted = driver_historical.sort_values(['Year', 'RoundNumber'], ascending=False)
                elif 'RoundNumber' in driver_historical.columns:
                    driver_historical_sorted = driver_historical.sort_values('RoundNumber', ascending=False)
                else:
                    driver_historical_sorted = driver_historical
                valid_team_names = driver_historical_sorted['TeamName'].dropna()
                if not valid_team_names.empty:
                    team_name = str(valid_team_names.iloc[0]).strip() if str(valid_team_names.iloc[0]).strip() else None
        
        # ALWAYS calculate constructor standing dynamically from completed races
        # This is critical for accuracy - never use stale values from driver_row
        constructor_standing = 10  # Default fallback
        constructor_points = 0
        
        if not completed_races.empty and 'Points' in completed_races.columns:
            # Get all races from current season up to this point
            season_races = completed_races[completed_races['Year'] == selected_year].copy()
            if season_races.empty:
                # If no races in current season, use most recent year from completed races
                latest_year = completed_races['Year'].max()
                season_races = completed_races[completed_races['Year'] == latest_year].copy()
            
            if not season_races.empty:
                # Remove rows with missing TeamName or Points for accurate calculation
                valid_season_races = season_races[
                    season_races['TeamName'].notna() & 
                    (season_races['TeamName'] != '') &
                    season_races['Points'].notna()
                ].copy()
                
                if not valid_season_races.empty and 'TeamName' in valid_season_races.columns:
                    # Calculate constructor points (sum of both drivers' points) by TeamName
                    constructor_points_dict = valid_season_races.groupby('TeamName')['Points'].sum().sort_values(ascending=False)
                    constructor_standings = {team: rank + 1 for rank, team in enumerate(constructor_points_dict.index)}
                    
                    # ALWAYS look up team from driver's races in season data (most reliable)
                    # Use MOST RECENT race to handle mid-season team switches
                    driver_season_races = valid_season_races[valid_season_races['DriverNumber'] == driver_num]
                    if not driver_season_races.empty:
                        # Sort by RoundNumber descending to get most recent race first
                        driver_season_races_sorted = driver_season_races.sort_values('RoundNumber', ascending=False)
                        driver_team = str(driver_season_races_sorted['TeamName'].iloc[0]).strip()
                        if driver_team in constructor_standings:
                            constructor_standing = constructor_standings[driver_team]
                            constructor_points = constructor_points_dict[driver_team]
                            team_name = driver_team  # Update team_name for later use
                        elif team_name and str(team_name).strip() in constructor_standings:
                            # Fallback to team_name we found earlier
                            constructor_standing = constructor_standings[str(team_name).strip()]
                            constructor_points = constructor_points_dict[str(team_name).strip()]
                        else:
                            # Team not found - this shouldn't happen
                            print(f"  Warning: Driver {driver_num} ({driver_name}) team '{driver_team}' not in standings. Available: {list(constructor_standings.keys())[:5]}")
                    elif team_name and str(team_name).strip() in constructor_standings:
                        # Driver not in season data but we have team_name from elsewhere
                        constructor_standing = constructor_standings[str(team_name).strip()]
                        constructor_points = constructor_points_dict[str(team_name).strip()]
                    else:
                        # Driver not found in season data at all
                        print(f"  Warning: Driver {driver_num} ({driver_name}) not found in season races")
                else:
                    # No valid TeamName data - this shouldn't happen if data collection is correct
                    print(f"  Warning: No valid TeamName data in season races for driver {driver_num}")
                    # Fallback: use driver points to estimate
                    driver_points = season_races[season_races['DriverNumber'] == driver_num]['Points'].sum()
                    all_driver_points = season_races.groupby('DriverNumber')['Points'].sum().sort_values(ascending=False)
                    if driver_num in all_driver_points.index:
                        driver_rank = list(all_driver_points.index).index(driver_num) + 1
                        constructor_standing = max(1, min(10, (driver_rank + 1) // 2))
                        constructor_points = driver_points * 2
            else:
                # No season races - use fallback
                print(f"  Warning: No season races found for driver {driver_num}, using default standing 10")
        else:
            # No completed races - use fallback
            print(f"  Warning: No completed races or Points column for driver {driver_num}, using default standing 10")
        
        # Calculate ConstructorTrackAvg: Constructor's average finish at this specific track
        # CRITICAL: Always use TeamName, never fall back to ConstructorStanding (unreliable)
        # Ensure team_name is set from most recent race
        if not team_name or not team_name.strip():
            # Try to get from most recent completed race
            if not driver_completed_races.empty and 'TeamName' in driver_completed_races.columns:
                driver_races_sorted = driver_completed_races.sort_values('RoundNumber', ascending=False)
                valid_team_names = driver_races_sorted['TeamName'].dropna()
                if not valid_team_names.empty:
                    team_name = str(valid_team_names.iloc[0]).strip()
        
        combined_historical = historical_df.copy()
        if not completed_races.empty:
            combined_historical = pd.concat([historical_df, completed_races], ignore_index=True)
        
        # Always use current team's track average (team_name should be set from most recent race)
        constructor_track_avg = 10.0  # Default
        if team_name and team_name.strip() and 'TeamName' in combined_historical.columns:
            constructor_track_races = combined_historical[
                (combined_historical['EventName'] == track_name) &
                (combined_historical['TeamName'] == team_name) &
                ((combined_historical['Year'] < selected_year) | 
                 ((combined_historical['Year'] == selected_year) & (combined_historical['RoundNumber'] < selected_round)))
            ]

            if not constructor_track_races.empty:
                pos_col = 'ActualPosition' if 'ActualPosition' in constructor_track_races.columns else 'Position'
                if pos_col in constructor_track_races.columns:
                    positions = constructor_track_races[pos_col].dropna()
                    if len(positions) > 0:
                        constructor_track_avg = positions.mean()
                    else:
                        constructor_track_avg = 10.0
                else:
                    constructor_track_avg = 10.0
            else:
                # Debug: team has no history at this track (only warn if team_name was set)
                if team_name and team_name.strip() and driver_name:
                    print(f"    DEBUG: {driver_name} ({driver_num}): Team={team_name} has no history at {track_name}, using default 10.0")
                constructor_track_avg = 10.0
        else:
            # Debug: team_name not available (this is a problem - should always be set)
            if driver_name:
                print(f"    WARNING: {driver_name} ({driver_num}): TeamName not available (team_name='{team_name}'), using default 10.0")
            constructor_track_avg = 10.0
        
        # Get track-specific historical average for this driver at this track
        # Default to 10.0 for rookies/drivers with no historical data
        hist_track_avg = track_avg_by_driver.get(str(driver_num), 
                                                  driver_row.get('HistoricalTrackAvgPosition', 10.0))
        if pd.isna(hist_track_avg):
            hist_track_avg = 10.0
        
        # Get driver's season-specific average grid position (or use most recent if no history)
        driver_grid_avg = grid_avg_by_driver.get(str(driver_num), np.nan)
        if pd.isna(driver_grid_avg):
            # Fallback: use most recent grid position, or historical average from driver_row
            driver_grid_avg = driver_row.get('GridPosition', np.nan)
            if pd.isna(driver_grid_avg):
                # Last resort: use driver's season-specific average from historical data (current season only)
                driver_season_races = historical_df[
                    (historical_df['DriverNumber'] == driver_num) &
                    (historical_df['Year'] == selected_year) &
                    (historical_df['RoundNumber'] < selected_round)
                ]
                if not driver_season_races.empty and 'GridPosition' in driver_season_races.columns:
                    valid_grid = driver_season_races['GridPosition'].dropna()
                    if len(valid_grid) > 0:
                        driver_grid_avg = valid_grid.mean()
        
        # Ensure we always have a valid AvgGridPosition (default to 10.5 = mid-field if no history)
        if pd.isna(driver_grid_avg):
            driver_grid_avg = 10.5  # Mid-field default for drivers with no grid history
        
        # Calculate TrackType (street circuit = 1, permanent = 0)
        street_circuits = [
            'Monaco', 'Singapore', 'Azerbaijan', 'Miami', 'Las Vegas',
            'Saudi Arabian'
        ]
        track_type = 1 if any(street in track_name for street in street_circuits) else 0
        
        # Use calculated cumulative features
        features = {
            'Year': selected_year,
            'EventName': track_name,
            'RoundNumber': selected_round,
            'SeasonPoints': season_points,  # Keep for backward compatibility
            'SeasonStanding': season_standing,  # Championship position (1 = leader, higher = worse)
            'SeasonAvgFinish': season_avg_finish,  # Calculated from all completed races
            'HistoricalTrackAvgPosition': hist_track_avg if not pd.isna(hist_track_avg) else 10.0,  # Track-specific average (default 10.0 for rookies)
            'ConstructorPoints': constructor_points,
            'ConstructorStanding': constructor_standing,
            'ConstructorTrackAvg': constructor_track_avg,  # Constructor's average finish at this track
            'GridPosition': driver_grid_avg if not pd.isna(driver_grid_avg) else 10.5,  # AvgGridPosition: driver's season-specific average grid position (default 10.5 if no history)
            'RecentForm': recent_form,  # Calculated from last 5 completed races
            'TrackType': track_type,  # Street circuit (1) or permanent (0)
            'DriverNumber': driver_num,
            'DriverName': driver_name,
            'TeamName': team_name if team_name else '',  # Team name for constructor matching
            'ActualPosition': np.nan  # Future race, no actual position
        }
        future_race_features.append(features)
    
    # Debug: Show team assignments for ConstructorTrackAvg verification
    print(f"\n  DEBUG: Team assignments for ConstructorTrackAvg calculation:")
    for feat in future_race_features:
        driver_name = feat.get('DriverName', 'Unknown')
        driver_num = feat.get('DriverNumber', 'Unknown')
        team = feat.get('TeamName', 'Unknown')
        constr_track_avg = feat.get('ConstructorTrackAvg', 10.0)
        print(f"    {driver_name} ({driver_num}): Team={team}, ConstructorTrackAvg={constr_track_avg:.2f}")
    
    return pd.DataFrame(future_race_features)


def recalculate_features_from_state(race_df: pd.DataFrame, previous_state_df: pd.DataFrame, 
                                    track_name: str, training_df: pd.DataFrame = None,
                                    test_df: pd.DataFrame = None, current_year: int = None,
                                    current_round: int = None) -> pd.DataFrame:
    """
    Recalculate ALL features for a future race based on the updated state from previous future race.
    This ensures features properly reflect all previous races including simulated future ones.
    """
    updated_df = race_df.copy()
    
    # Get track-specific historical averages
    historical_df = training_df if training_df is not None and not training_df.empty else None
    track_avg_by_driver = {}
    if historical_df is not None:
        track_historical_data = historical_df[historical_df['EventName'] == track_name]
        for driver_num in track_historical_data['DriverNumber'].unique():
            driver_track_races = track_historical_data[track_historical_data['DriverNumber'] == driver_num]
            if 'ActualPosition' in driver_track_races.columns:
                valid_positions = driver_track_races['ActualPosition'].dropna()
                if len(valid_positions) > 0:
                    track_avg_by_driver[str(driver_num)] = valid_positions.mean()
    
    # Calculate TrackType
    street_circuits = [
        'Monaco', 'Singapore', 'Azerbaijan', 'Miami', 'Las Vegas',
        'Saudi Arabian'
    ]
    track_type = 1 if any(street in track_name for street in street_circuits) else 0
    
    # Update features from previous state for each driver
    for idx, row in updated_df.iterrows():
        driver_num = row['DriverNumber']
        prev_driver_data = previous_state_df[previous_state_df['DriverNumber'] == driver_num]
        
        if not prev_driver_data.empty:
            prev_row = prev_driver_data.iloc[0]
            # Update all features from previous state
            updated_df.at[idx, 'SeasonPoints'] = prev_row.get('SeasonPoints', row.get('SeasonPoints', 0))
            # Calculate SeasonStanding from test_df if available
            if test_df is not None and current_year is not None and current_round is not None:
                # Get all races from current season up to current round
                season_races = test_df[
                    (test_df['Year'] == current_year) &
                    (test_df['RoundNumber'] < current_round) &
                    (test_df['Points'].notna())
                ].copy()
                
                if not season_races.empty:
                    # Calculate points per driver
                    driver_points_dict = season_races.groupby('DriverNumber')['Points'].sum().sort_values(ascending=False)
                    
                    # Find this driver's position (1 = most points)
                    if driver_num in driver_points_dict.index:
                        driver_total_points = driver_points_dict[driver_num]
                        position = (driver_points_dict > driver_total_points).sum() + 1
                        updated_df.at[idx, 'SeasonStanding'] = position
                    else:
                        updated_df.at[idx, 'SeasonStanding'] = len(driver_points_dict) + 1 if len(driver_points_dict) > 0 else 20
                else:
                    updated_df.at[idx, 'SeasonStanding'] = prev_row.get('SeasonStanding', row.get('SeasonStanding', 20))
            else:
                updated_df.at[idx, 'SeasonStanding'] = prev_row.get('SeasonStanding', row.get('SeasonStanding', 20))
            updated_df.at[idx, 'SeasonAvgFinish'] = prev_row.get('SeasonAvgFinish', row.get('SeasonAvgFinish', np.nan))
            updated_df.at[idx, 'ConstructorPoints'] = prev_row.get('ConstructorPoints', row.get('ConstructorPoints', 0))
            updated_df.at[idx, 'RecentForm'] = prev_row.get('RecentForm', row.get('RecentForm', np.nan))
            # Preserve TeamName if available
            team_name = None
            if 'TeamName' in prev_row and pd.notna(prev_row.get('TeamName')) and str(prev_row.get('TeamName')).strip():
                team_name = str(prev_row['TeamName']).strip()
                updated_df.at[idx, 'TeamName'] = team_name
            elif 'TeamName' in row and pd.notna(row.get('TeamName')) and str(row.get('TeamName')).strip():
                team_name = str(row['TeamName']).strip()
                updated_df.at[idx, 'TeamName'] = team_name
            
            # ALWAYS recalculate ConstructorStanding dynamically from test_df if available
            # This ensures accuracy even if previous state has stale values
            constructor_standing = 10  # Default
            if test_df is not None and current_year is not None and current_round is not None:
                # Get completed races up to current round
                completed_races = test_df[
                    (test_df['Year'] == current_year) & 
                    (test_df['RoundNumber'] < current_round)
                ].copy()
                
                if not completed_races.empty and 'Points' in completed_races.columns:
                    # Remove rows with missing TeamName or Points
                    valid_races = completed_races[
                        completed_races['TeamName'].notna() & 
                        (completed_races['TeamName'].astype(str).str.strip() != '') &
                        completed_races['Points'].notna()
                    ].copy()
                    
                    if not valid_races.empty and 'TeamName' in valid_races.columns:
                        # Calculate constructor standings
                        constructor_points_dict = valid_races.groupby('TeamName')['Points'].sum().sort_values(ascending=False)
                        constructor_standings = {team: rank + 1 for rank, team in enumerate(constructor_points_dict.index)}
                        
                        # ALWAYS look up driver's team from their races (most reliable)
                        # Use MOST RECENT race to handle mid-season team switches
                        driver_races = valid_races[valid_races['DriverNumber'] == driver_num]
                        if not driver_races.empty:
                            # Sort by RoundNumber descending to get most recent race first
                            driver_races_sorted = driver_races.sort_values('RoundNumber', ascending=False)
                            driver_team = str(driver_races_sorted['TeamName'].iloc[0]).strip()
                            if driver_team in constructor_standings:
                                constructor_standing = constructor_standings[driver_team]
                                team_name = driver_team  # Update for later use
                                updated_df.at[idx, 'TeamName'] = team_name
                            elif team_name and str(team_name).strip() in constructor_standings:
                                # Fallback to team_name we found earlier
                                constructor_standing = constructor_standings[str(team_name).strip()]
                            else:
                                # Team not found - use fallback
                                constructor_standing = prev_row.get('ConstructorStanding', 10)
                        elif team_name and str(team_name).strip() in constructor_standings:
                            # Driver not in races but we have team_name from elsewhere
                            constructor_standing = constructor_standings[str(team_name).strip()]
                        else:
                            # Driver not found - use fallback
                            constructor_standing = prev_row.get('ConstructorStanding', 10)
                    else:
                        # No valid data - use previous value
                        constructor_standing = prev_row.get('ConstructorStanding', 10)
                else:
                    # No completed races - use previous value
                    constructor_standing = prev_row.get('ConstructorStanding', 10)
            else:
                # No test_df or year/round info - use previous value
                constructor_standing = prev_row.get('ConstructorStanding', 10)
            
            updated_df.at[idx, 'ConstructorStanding'] = constructor_standing
            
            # AvgGridPosition should be preserved (season-specific average grid position)
            if pd.isna(row.get('GridPosition', np.nan)) and not pd.isna(prev_row.get('GridPosition', np.nan)):
                updated_df.at[idx, 'GridPosition'] = prev_row.get('GridPosition')
            elif pd.isna(row.get('GridPosition', np.nan)):
                updated_df.at[idx, 'GridPosition'] = 10.5  # Default if still missing
            
            # HistoricalTrackAvgPosition - use track-specific if available
            hist_track_avg = track_avg_by_driver.get(str(driver_num), prev_row.get('HistoricalTrackAvgPosition', 10.0))
            if pd.isna(hist_track_avg):
                hist_track_avg = 10.0
            updated_df.at[idx, 'HistoricalTrackAvgPosition'] = hist_track_avg
            
            # ConstructorTrackAvg - calculate constructor's average at this track
            # CRITICAL: Always use TeamName, never fall back to ConstructorStanding (unreliable)
            # Ensure team_name is set from most recent race or previous state
            if not team_name or not team_name.strip():
                # Try to get from previous state
                if 'TeamName' in prev_row and pd.notna(prev_row.get('TeamName')):
                    team_name = str(prev_row['TeamName']).strip()
                elif 'TeamName' in row and pd.notna(row.get('TeamName')):
                    team_name = str(row['TeamName']).strip()
            
            constructor_track_avg = 10.0  # Default
            if team_name and team_name.strip() and historical_df is not None and 'TeamName' in historical_df.columns:
                constructor_track_races = historical_df[
                    (historical_df['EventName'] == track_name) &
                    (historical_df['TeamName'] == team_name)
                ]

                if not constructor_track_races.empty:
                    pos_col = 'ActualPosition' if 'ActualPosition' in constructor_track_races.columns else 'Position'
                    if pos_col in constructor_track_races.columns:
                        positions = constructor_track_races[pos_col].dropna()
                        if len(positions) > 0:
                            constructor_track_avg = positions.mean()
                        else:
                            constructor_track_avg = 10.0
                    else:
                        constructor_track_avg = 10.0
                else:
                    # Debug: team has no history at this track (only warn if team_name was set)
                    if team_name and team_name.strip():
                        driver_name = row.get('DriverName', f"Driver {driver_num}")
                        print(f"    DEBUG: {driver_name} ({driver_num}): Team={team_name} has no history at {track_name}, using default 10.0")
                    constructor_track_avg = 10.0
            else:
                # Debug: team_name not available (this is a problem - should always be set)
                driver_name = row.get('DriverName', f"Driver {driver_num}")
                print(f"    WARNING: {driver_name} ({driver_num}): TeamName not available (team_name='{team_name}'), using default 10.0")
                constructor_track_avg = 10.0
            
            updated_df.at[idx, 'ConstructorTrackAvg'] = constructor_track_avg
            
            # Update TrackType (track-specific feature)
            updated_df.at[idx, 'TrackType'] = track_type
            
    
    return updated_df


def update_state_with_actual_results(state_df: pd.DataFrame, race_results_df: pd.DataFrame, 
                                     test_df: pd.DataFrame = None, current_year: int = None, 
                                     current_round: int = None) -> pd.DataFrame:
    """
    Update state DataFrame with actual race results.
    This updates season points and other features based on actual finishing positions.
    Recalculates SeasonAvgFinish and RecentForm from all completed races for accuracy.
    """
    updated_state = state_df.copy()
    
    # Update each driver's features based on their actual position
    # NOTE: We rely entirely on API Points column - no manual calculation
    for _, result_row in race_results_df.iterrows():
        driver_num = result_row.get('DriverNumber')
        if driver_num is not None:
            state_idx = updated_state[updated_state['DriverNumber'] == driver_num].index
            if len(state_idx) > 0:
                idx = state_idx[0]
                
                # Get actual position from results
                if 'ActualPosition' in result_row and not pd.isna(result_row['ActualPosition']):
                    actual_pos = int(result_row['ActualPosition'])
                elif 'Position' in result_row and not pd.isna(result_row['Position']):
                    actual_pos = int(result_row['Position'])
                else:
                    continue  # Skip if no actual position
                
                # Update ActualPosition
                updated_state.at[idx, 'ActualPosition'] = actual_pos
                
                # Recalculate SeasonPoints, SeasonAvgFinish and RecentForm from all completed races (if test_df available)
                if test_df is not None and current_year is not None and current_round is not None:
                    # Convert driver_num to same type as test_df for matching
                    # Handle both string and int types
                    test_df_driver_num_type = test_df['DriverNumber'].dtype
                    if test_df_driver_num_type == 'object' or test_df['DriverNumber'].dtype.name == 'object':
                        # String type - convert driver_num to string
                        driver_num_match = str(driver_num)
                    else:
                        # Numeric type - try to convert driver_num to int/float
                        try:
                            driver_num_match = int(driver_num) if isinstance(driver_num, (int, float, str)) else driver_num
                        except (ValueError, TypeError):
                            driver_num_match = driver_num
                    
                    # Get all completed races for this driver up to and including current race
                    driver_completed_races = test_df[
                        (test_df['DriverNumber'] == driver_num_match) &
                        (test_df['Year'] == current_year) &
                        (test_df['RoundNumber'] <= current_round) &
                        (test_df['ActualPosition'].notna())
                    ].copy()
                    
                    if not driver_completed_races.empty:
                        # Recalculate SeasonPoints from ALL completed races
                        # RELY ONLY ON API Points COLUMN - no manual calculation
                        if 'Points' in driver_completed_races.columns:
                            # Use Points column directly from API (source of truth)
                            # Ensure Points are numeric and handle NaN values
                            points_series = pd.to_numeric(driver_completed_races['Points'], errors='coerce')
                            total_points = points_series.sum()
                            
                            # Validate: points should be reasonable (0-600 for a full season)
                            if total_points < 0 or total_points > 600:
                                print(f"  Warning: Driver {driver_num} has suspicious total points: {total_points} after round {current_round}")
                                print(f"    Races included: {len(driver_completed_races)}")
                                print(f"    Points breakdown: {driver_completed_races[['RoundNumber', 'EventName', 'Points']].to_dict('records')}")
                            
                            updated_state.at[idx, 'SeasonPoints'] = total_points
                            
                            # Calculate SeasonStanding: championship position based on points (1 = leader, higher = worse)
                            # Get all races from current season up to and including current round
                            season_races = test_df[
                                (test_df['Year'] == current_year) &
                                (test_df['RoundNumber'] <= current_round) &
                                (test_df['Points'].notna())
                            ].copy()
                            
                            if not season_races.empty:
                                # Calculate points per driver
                                driver_points_dict = season_races.groupby('DriverNumber')['Points'].sum().sort_values(ascending=False)
                                
                                # Find this driver's position (1 = most points)
                                if driver_num_match in driver_points_dict.index:
                                    driver_total_points = driver_points_dict[driver_num_match]
                                    # Count how many drivers have more points
                                    position = (driver_points_dict > driver_total_points).sum() + 1
                                    updated_state.at[idx, 'SeasonStanding'] = position
                                else:
                                    # Driver has no points yet - assign worst position
                                    updated_state.at[idx, 'SeasonStanding'] = len(driver_points_dict) + 1 if len(driver_points_dict) > 0 else 20
                        else:
                            # Points column missing - keep current points as fallback
                            # This should rarely happen if data collection is correct
                            print(f"  Warning: No Points column found for driver {driver_num} in test_df, keeping current points")
                        
                        # Calculate true SeasonAvgFinish from all completed races
                        if 'ActualPosition' in driver_completed_races.columns:
                            valid_positions = driver_completed_races['ActualPosition'].dropna()
                            if len(valid_positions) > 0:
                                updated_state.at[idx, 'SeasonAvgFinish'] = valid_positions.mean()
                        
                        # Calculate RecentForm from last 5 races
                        driver_races_sorted = driver_completed_races.sort_values('RoundNumber', ascending=False)
                        last_5_races = driver_races_sorted.head(5)
                        valid_recent_positions = last_5_races['ActualPosition'].dropna()
                        if len(valid_recent_positions) > 0:
                            updated_state.at[idx, 'RecentForm'] = valid_recent_positions.mean()
                        else:
                            updated_state.at[idx, 'RecentForm'] = np.nan
                        
                    else:
                        # Fallback: use approximation if we can't recalculate from test_df
                        # Try to get points from race_results_df if available
                        if 'Points' in result_row and not pd.isna(result_row['Points']):
                            # Use Points from current race result (API data)
                            points_from_race = result_row['Points']
                            current_points = updated_state.at[idx, 'SeasonPoints']
                            updated_state.at[idx, 'SeasonPoints'] = current_points + points_from_race
                        else:
                            # No Points column available - keep current points
                            # This should rarely happen if data collection is correct
                            print(f"  Warning: No Points column found for driver {driver_num} in race results, keeping current points")
                        
                        current_avg = updated_state.at[idx, 'SeasonAvgFinish']
                        if pd.isna(current_avg):
                            updated_state.at[idx, 'SeasonAvgFinish'] = float(actual_pos)
                        else:
                            # Approximate: assume this is race N, update average
                            updated_state.at[idx, 'SeasonAvgFinish'] = (current_avg * 0.9 + actual_pos * 0.1)
                        
                        current_recent_form = updated_state.at[idx, 'RecentForm']
                        if pd.isna(current_recent_form):
                            updated_state.at[idx, 'RecentForm'] = float(actual_pos)
                        else:
                            updated_state.at[idx, 'RecentForm'] = (current_recent_form * 0.8 + actual_pos * 0.2)
                        
                else:
                    # Fallback: use approximation if test_df not available
                    # Try to get points from race_results_df if available
                    if 'Points' in result_row and not pd.isna(result_row['Points']):
                        # Use Points from current race result (API data)
                        points_from_race = result_row['Points']
                        current_points = updated_state.at[idx, 'SeasonPoints']
                        updated_state.at[idx, 'SeasonPoints'] = current_points + points_from_race
                    else:
                        # No Points column available - keep current points
                        # This should rarely happen if data collection is correct
                        print(f"  Warning: No Points column found for driver {driver_num} in race results (fallback), keeping current points")
                    
                    current_avg = updated_state.at[idx, 'SeasonAvgFinish']
                    if pd.isna(current_avg):
                        updated_state.at[idx, 'SeasonAvgFinish'] = float(actual_pos)
                    else:
                        updated_state.at[idx, 'SeasonAvgFinish'] = (current_avg * 0.9 + actual_pos * 0.1)
                    
                    current_recent_form = updated_state.at[idx, 'RecentForm']
                    if pd.isna(current_recent_form):
                        updated_state.at[idx, 'RecentForm'] = float(actual_pos)
                    else:
                        updated_state.at[idx, 'RecentForm'] = (current_recent_form * 0.8 + actual_pos * 0.2)
                    
    
    return updated_state


def update_state_with_predictions(state_df: pd.DataFrame, predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Update state DataFrame with predictions as simulated race results.
    This allows progressive feature calculation across future races.
    Simulates what features would be after this race based on predicted positions.
    """
    updated_state = state_df.copy()
    
    # F1 points system (position -> points)
    points_system = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
    
    # Sort predictions by predicted position to get actual race order
    sorted_predictions = predictions_df.sort_values('PredictedPosition').copy()
    
    # Update each driver's features based on their predicted position
    for rank, (_, pred_row) in enumerate(sorted_predictions.iterrows(), 1):
        driver_num = pred_row.get('DriverNumber')
        if driver_num is not None:
            state_idx = updated_state[updated_state['DriverNumber'] == driver_num].index
            if len(state_idx) > 0:
                idx = state_idx[0]
                pred_pos = pred_row.get('PredictedPosition', rank)
                actual_pos = int(round(pred_pos))  # Round to nearest integer for simulation
                
                # Update ActualPosition
                updated_state.at[idx, 'ActualPosition'] = actual_pos
                
                # Update SeasonPoints (add points from this race)
                current_points = updated_state.at[idx, 'SeasonPoints']
                points_earned = points_system.get(actual_pos, 0)
                updated_state.at[idx, 'SeasonPoints'] = current_points + points_earned
                
                # Note: SeasonStanding will be recalculated after all drivers are updated
                
                # Update SeasonAvgFinish (recalculate average)
                # Get all previous positions including this one
                current_avg = updated_state.at[idx, 'SeasonAvgFinish']
                if pd.isna(current_avg):
                    updated_state.at[idx, 'SeasonAvgFinish'] = float(actual_pos)
                else:
                    # Approximate: assume this is race N, update average
                    # This is simplified - ideally we'd track all previous positions
                    updated_state.at[idx, 'SeasonAvgFinish'] = (current_avg * 0.9 + actual_pos * 0.1)
                
                # Update RecentForm (last 5 races average)
                # Simplified: shift the average
                current_recent_form = updated_state.at[idx, 'RecentForm']
                if pd.isna(current_recent_form):
                    updated_state.at[idx, 'RecentForm'] = float(actual_pos)
                else:
                    # Approximate rolling average (simplified)
                    updated_state.at[idx, 'RecentForm'] = (current_recent_form * 0.8 + actual_pos * 0.2)
    
    # Recalculate SeasonStanding for all drivers based on updated SeasonPoints
    # Sort drivers by SeasonPoints (descending) and assign positions
    if 'SeasonPoints' in updated_state.columns:
        # Get all drivers with valid points
        valid_drivers = updated_state[updated_state['SeasonPoints'].notna()].copy()
        if not valid_drivers.empty:
            # Sort by points (descending) to get standings
            valid_drivers = valid_drivers.sort_values('SeasonPoints', ascending=False)
            
            # Assign positions (1 = most points)
            position = 1
            prev_points = None
            for idx, row in valid_drivers.iterrows():
                current_points = row['SeasonPoints']
                # If points are the same as previous driver, they share the position
                if prev_points is not None and current_points < prev_points:
                    position = len(valid_drivers[valid_drivers['SeasonPoints'] > current_points]) + 1
                updated_state.at[idx, 'SeasonStanding'] = position
                prev_points = current_points
    
    # Update ConstructorPoints and ConstructorStanding based on team points
    # Group by constructor (simplified - would need TeamName column)
    # For now, we'll skip this as it requires constructor mapping
    
    return updated_state

