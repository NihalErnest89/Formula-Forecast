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


def _canonical_driver_num(x):
    """Normalize driver number so 4, 4.0, '4' all become '4' (avoids double-counting wins)."""
    if pd.isna(x):
        return x
    try:
        return str(int(float(x)))
    except (ValueError, TypeError):
        return str(x)


# Directory for raw per-season result snapshots (enables incremental collection:
# already-collected rounds are read from disk and only NEW races hit the API)
RAW_DATA_DIR = Path(__file__).parent / 'data' / 'raw'


def _has_valid_results(df: pd.DataFrame) -> bool:
    """True if a race result block has real finishing-position data.

    FastF1 returns a full driver list even when the Ergast/Jolpica results API
    is rate-limited (429) or has no data yet, leaving Position/Status all-NaN.
    Such placeholder rows must NOT be treated as a collected race.
    """
    if df is None or df.empty or 'Position' not in df.columns:
        return False
    return df['Position'].notna().any()


def _event_is_past(event) -> bool:
    """True if the event has already taken place (so results should exist)."""
    ev_date = event.get('EventDate', None)
    if pd.isna(ev_date):
        return True  # Unknown date: attempt the fetch, worst case it fails gracefully
    ts = pd.Timestamp(ev_date)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts <= pd.Timestamp.now()


def get_season_data(year: int, max_retries: int = 3, force_refresh: bool = False) -> Tuple[pd.DataFrame, int]:
    """
    Get all race data for a given season (incrementally).

    Rounds already saved in data/raw/season_<year>.csv are loaded from disk;
    only new (not yet collected, already completed) races are fetched from the
    Fast F1 API. Pre-season testing events and future races are skipped.

    Args:
        year: Season year (e.g., 2024)
        max_retries: Maximum number of retry attempts for API calls
        force_refresh: If True, ignore the local snapshot and refetch everything

    Returns:
        Tuple of (DataFrame with race results, number of NEWLY fetched races).
        The new-race count lets callers skip the expensive feature
        reorganization when nothing changed.
    """
    # Load existing snapshot for incremental collection
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DATA_DIR / f'season_{year}.csv'
    existing = pd.DataFrame()
    if raw_path.exists() and not force_refresh:
        try:
            existing = pd.read_csv(raw_path)
        except Exception as e:
            print(f"  Warning: Could not read snapshot {raw_path} ({e}); refetching season.")
            existing = pd.DataFrame()
    # A round counts as "done" only if its snapshot rows have real finishing
    # positions. Rounds saved with empty results (API was unavailable at fetch
    # time) are deliberately NOT marked done, so they get refetched next run.
    done_rounds = set()
    if not existing.empty and 'RoundNumber' in existing.columns:
        for rnd, grp in existing.groupby('RoundNumber'):
            if _has_valid_results(grp):
                done_rounds.add(rnd)

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
                return existing, 0

    if schedule is None or schedule.empty:
        print(f"  Warning: No schedule found for {year}")
        return existing, 0

    # Skip pre-season testing events: they have RoundNumber 0 and their "results"
    # are meaningless for race prediction (previously they were collected and even
    # assigned fabricated race points, contaminating the training data)
    schedule = schedule[schedule['RoundNumber'] >= 1]

    all_races = []
    skipped_cached = 0
    skipped_future = 0
    skipped_no_results = 0

    for _, event in schedule.iterrows():
        # Incremental: skip rounds we already have on disk
        if event['RoundNumber'] in done_rounds:
            skipped_cached += 1
            continue
        # Skip races that haven't happened yet (no results to fetch)
        if not _event_is_past(event):
            skipped_future += 1
            continue
        try:
            # Get race session (results only - laps/telemetry/weather not needed)
            session = fastf1.get_session(year, event['EventName'], 'R')
            session.load(laps=False, telemetry=False, weather=False, messages=False)
            
            # Get race results
            results = session.results
            if results is not None and not results.empty:
                results['Year'] = year
                results['EventName'] = event['EventName']
                results['RoundNumber'] = event['RoundNumber']
                
                # Calculate race points from finishing position (more reliable than Points column)
                # F1 points system: 1st=25, 2nd=18, 3rd=15, 4th=12, 5th=10, 6th=8, 7th=6, 8th=4, 9th=2, 10th=1
                points_system = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
                
                # Get race position (use Position column, or Status if DNF)
                if 'Position' in results.columns:
                    results['RacePoints'] = results['Position'].map(points_system).fillna(0)
                else:
                    # Fallback: use Points column if Position not available
                    results['RacePoints'] = results['Points'].fillna(0)
                
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
                    # Total points = race points (calculated from position) + sprint points
                    results['TotalEventPoints'] = results['RacePoints'].fillna(0) + results['SprintPoints']
                else:
                    results['SprintPoints'] = 0
                    # Total points = race points (calculated from position)
                    results['TotalEventPoints'] = results['RacePoints'].fillna(0)
                
                # Try to get qualifying/starting grid position
                # Check if GridPosition column exists, if not try to get from qualifying session
                if 'GridPosition' not in results.columns:
                    try:
                        # Try to get qualifying session (results only)
                        qual_session = fastf1.get_session(year, event['EventName'], 'Q')
                        qual_session.load(laps=False, telemetry=False, weather=False, messages=False)
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
                
                results_df = pd.DataFrame(results)
                # Only keep races that actually have finishing-position data.
                # Placeholder rows (API unavailable) are dropped so they are not
                # persisted and get retried on the next run.
                if _has_valid_results(results_df):
                    all_races.append(results_df)
                else:
                    skipped_no_results += 1
                    print(f"  Note: {event['EventName']} {year} (R{event['RoundNumber']}) "
                          f"returned no finishing data yet - will retry next run")
        except Exception as e:
            print(f"  Warning: Error loading race {event.get('EventName', 'Unknown')} {year}: {e}")
            continue

    if skipped_cached or skipped_future or skipped_no_results or all_races:
        print(f"  {year}: {len(all_races)} new races fetched, "
              f"{skipped_cached} loaded from snapshot, {skipped_future} future races skipped"
              + (f", {skipped_no_results} awaiting results" if skipped_no_results else ""))

    # Drop any stale placeholder rows for rounds we just refetched with real data
    refetched_rounds = {r for df in all_races for r in df['RoundNumber'].unique()}
    if not existing.empty and refetched_rounds and 'RoundNumber' in existing.columns:
        existing = existing[~existing['RoundNumber'].isin(refetched_rounds)]

    # Merge newly fetched races with the existing snapshot and persist
    frames = ([existing] if not existing.empty else []) + all_races
    if not frames:
        return pd.DataFrame(), 0

    season_df = pd.concat(frames, ignore_index=True)

    # Normalize DriverNumber to a canonical string: fresh fastf1 results carry it
    # as str ('44') but the CSV snapshot round-trip turns it into int64 (44),
    # which silently breaks downstream string comparisons (e.g. career win counts)
    if 'DriverNumber' in season_df.columns:
        season_df['DriverNumber'] = season_df['DriverNumber'].apply(_canonical_driver_num)

    if all_races:  # Only rewrite the snapshot when something new was fetched
        try:
            season_df.to_csv(raw_path, index=False)
            print(f"  Snapshot updated: {raw_path}")
        except Exception as e:
            print(f"  Warning: Could not save snapshot {raw_path}: {e}")
    return season_df, len(all_races)


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
            track_avg_dict[str(driver_num)] = 10.0  # Default for rookies
    
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


def _is_dnf(status, position_text=''):
    """True when the driver did NOT finish the race (retired, crashed, mechanical
    failure, disqualified, did not start, withdrew...).

    FastF1's Status is 'Finished', 'Lapped' or '+N Lap(s)' for a driver who saw the
    flag and a failure description otherwise ('Retired', 'Collision', 'Engine',
    'Disqualified', ...). The old check looked for the words DNF / DSQ / NC, which
    those descriptions never contain, so only 22 of ~590 retirements were flagged
    and the 'finishers only' filtering barely filtered anything."""
    if pd.notna(status) and str(status).strip():
        s = str(status).strip().lower()
        return not (s in ('finished', 'lapped') or s.startswith('+'))
    if pd.notna(position_text) and str(position_text).strip():
        # no status text: Ergast-style classified position -- R retired, D disqualified,
        # E excluded, W withdrew, F failed to qualify, N not classified
        return str(position_text).strip().upper() in ('R', 'D', 'E', 'W', 'F', 'N')
    return False


def _pitlane_to_back_of_grid(df: pd.DataFrame) -> pd.DataFrame:
    """FastF1 reports a pit-lane start as grid position 0. Everything downstream
    ranks or averages grid slots, so 0 would read as "ahead of pole" (and drag a
    driver's season-average grid toward the front). Treat it as the back of
    that race's grid instead."""
    if df.empty or 'GridPosition' not in df.columns:
        return df
    df = df.copy()
    field_size = df.groupby(['Year', 'RoundNumber'])['DriverNumber'].transform('count')
    df['GridPosition'] = df['GridPosition'].where(df['GridPosition'] != 0, field_size)
    return df


_PRE2018_WINS_PATH = Path(__file__).parent / 'data' / 'pre2018_wins.json'
_pre2018_wins_cache = None


def _pre2018_wins(driver_key, year):
    """Wins BEFORE 2018 for a driver, which the results data (2018 on) cannot see.

    Returns (career_wins_before_2018, wins_before_2018_that_fall_inside_the_3_year_window_of_`year`).
    The window part only matters for 2018 (covers 2016-17) and 2019 (covers 2017).
    Table built once by fetch_pre2018_wins.py; keyed by the same abbreviation as the features."""
    global _pre2018_wins_cache
    if _pre2018_wins_cache is None:
        if _PRE2018_WINS_PATH.exists():
            _pre2018_wins_cache = json.loads(_PRE2018_WINS_PATH.read_text(encoding='utf-8'))
        else:
            print(f"  WARNING: {_PRE2018_WINS_PATH} missing -- CareerWins will undercount "
                  "anyone who won before 2018. Run: python fetch_pre2018_wins.py")
            _pre2018_wins_cache = {}
    d = _pre2018_wins_cache.get(str(driver_key))
    if not d:
        return 0, 0
    last3 = sum(n for y, n in d['by_year'].items() if int(y) >= year - 2)
    return d['total'], last3


def _key_drivers_by_identity(df: pd.DataFrame) -> pd.DataFrame:
    """A driver number is not a driver. Champions swap to #1 and back, drivers
    move numbers (Verstappen 33 -> 1 -> 3, Norris 4 -> 1), and numbers get
    reused (Ricciardo's #3 became Verstappen's, Vettel's #5 became Bortoleto's).
    Every cross-season lookup below filters on DriverNumber, so a driver's
    wins/form/track history was split at each change and a number's new holder
    inherited the previous holder's record.

    While features are built, key every lookup on the driver's abbreviation
    instead. The real number is kept in _RealNumber and written back to the
    output tables, so downstream code still sees the number it always did."""
    if df.empty or 'Abbreviation' not in df.columns:
        return df
    df = df.copy()
    df['_RealNumber'] = df['DriverNumber']
    df['DriverNumber'] = df['Abbreviation'].fillna(df['DriverNumber'].astype(str))
    return df


def organize_data(training_years: List[int], test_years: List[int],
                  force_refresh: bool = False,
                  force_reorganize: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Collect and organize F1 data into features and labels.

    Args:
        training_years: List of years for training data (e.g., [2020, 2021, 2022, 2023, 2024, 2025])
        test_years: List of years for test data (e.g., [2026])
        force_refresh: If True, ignore local season snapshots and refetch everything
        force_reorganize: If True, rebuild feature CSVs even when no new races
            were fetched (use after changing feature engineering code)

    Returns:
        Tuple of (training_data, test_data) DataFrames, or (None, None) when no
        new races were fetched and the existing feature CSVs are already current
        (the expensive feature organization step is skipped).
    """
    total_new_races = 0
    new_training_races = 0

    print("Collecting training data...")
    training_races = []

    # Collect training data from past seasons
    successful_years = []
    for year in training_years:
        print(f"  Loading season {year}...")
        season_data, n_new = get_season_data(year, force_refresh=force_refresh)
        total_new_races += n_new
        new_training_races += n_new
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
    
    # Collect test data (multiple years so predict can offer both 2025 and 2026)
    test_races_list = []
    for ty in test_years:
        print(f"Collecting test data for {ty}...")
        season_df, n_new = get_season_data(ty, force_refresh=force_refresh)
        total_new_races += n_new
        if not season_df.empty:
            test_races_list.append(season_df)
    test_data = pd.concat(test_races_list, ignore_index=True) if test_races_list else pd.DataFrame()

    # Done here (not at fetch time) so cached season snapshots get it too.
    all_training_data = _pitlane_to_back_of_grid(all_training_data)
    test_data = _pitlane_to_back_of_grid(test_data)
    all_training_data = _key_drivers_by_identity(all_training_data)
    test_data = _key_drivers_by_identity(test_data)

    # Skip the expensive feature-organization pass when nothing new was fetched
    # and the feature CSVs already exist (they can only depend on the snapshots)
    output_csvs_exist = (Path('data') / 'training_data.csv').exists() and (Path('data') / 'test_data.csv').exists()
    if total_new_races == 0 and output_csvs_exist and not force_reorganize and not force_refresh:
        print("\nNo new races fetched - existing feature CSVs are current, skipping feature organization.")
        print("(Use 'python collect_data.py --reorganize' to force a rebuild after feature-code changes.)")
        return None, None

    # A training-season row's features depend only on training-season races. If
    # none of those changed, the existing training CSV is still exactly right,
    # so reuse it and rebuild only the test seasons. Mid-season every new race
    # is a test-season race, and the ~3,400 training rows are most of the work.
    # (The test loop below reads only the raw results, never anything the
    # training loop builds, so skipping the training loop is safe.)
    reuse_training = (new_training_races == 0 and output_csvs_exist
                      and not force_reorganize and not force_refresh)

    # Organize features and labels
    print(f"Organizing features and labels ({total_new_races} new races)...")
    if reuse_training:
        print("  Training seasons unchanged - reusing data/training_data.csv, rebuilding test seasons only")

    training_features = []
    test_features = []

    # Process training data
    for year in ([] if reuse_training else training_years):
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
                prev_year_data = all_training_data[all_training_data['Year'] == year - 1]
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
            
            # If no track-specific history, default to 10.0 for rookies (don't use overall average)
            if pd.isna(hist_track_avg):
                hist_track_avg = 10.0  # Default for rookies at this track
            
            # Store actual grid position from this race before calculating average
            actual_grid_position = race.get('GridPosition', np.nan)
            if pd.isna(actual_grid_position):
                # Try alternative column names
                actual_grid_position = race.get('StartingGrid', race.get('Grid', np.nan))
            
            # Get starting grid position - use AVERAGE grid position instead of actual
            # This matches what we'll use for future race predictions and eliminates train/test mismatch
            # Using season-specific average (only races from current season before current round)
            grid_position = calculate_average_grid_position(all_training_data, driver_num, year, round_num, season_specific=True)
            
            # Fallback: if no historical data, use actual grid position from this race
            if pd.isna(grid_position):
                grid_position = actual_grid_position
            
            # Recent form (last 5 races average finish) - captures current momentum
            if round_num == 1:
                # First race: use previous season's recent form
                prev_year_data = all_training_data[all_training_data['Year'] == year - 1]
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
                prev_year_data = all_training_data[all_training_data['Year'] == year - 1]
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
            
            # 2. CareerWins: lifetime wins prior to this race (all seasons in training set)
            prior_races = all_training_data[
                (all_training_data['Year'] < year) |
                ((all_training_data['Year'] == year) & (all_training_data['RoundNumber'] < round_num))
            ].copy()
            if not prior_races.empty:
                if 'ActualPosition' in prior_races.columns:
                    wins_col = 'ActualPosition'
                elif 'Position' in prior_races.columns:
                    wins_col = 'Position'
                else:
                    wins_col = None
                
                if wins_col is not None:
                    driver_prior = prior_races[prior_races['DriverNumber'] == driver_num]
                    career_wins = (driver_prior[wins_col] == 1).sum()
                else:
                    career_wins = 0
            else:
                career_wins = 0
            
            # 2b. WinsLast3Years: wins in the last 3 calendar years (recency-weighted "on fire" signal)
            recent_start_year = max(all_training_data['Year'].min(), year - 2)
            prior_races_3y = all_training_data[
                (all_training_data['Year'] >= recent_start_year) &
                ((all_training_data['Year'] < year) |
                 ((all_training_data['Year'] == year) & (all_training_data['RoundNumber'] < round_num)))
            ].copy()
            if not prior_races_3y.empty:
                if 'ActualPosition' in prior_races_3y.columns:
                    wcol = 'ActualPosition'
                elif 'Position' in prior_races_3y.columns:
                    wcol = 'Position'
                else:
                    wcol = None
                if wcol is not None:
                    driver_prior_3y = prior_races_3y[prior_races_3y['DriverNumber'] == driver_num]
                    wins_last_3_years = (driver_prior_3y[wcol] == 1).sum()
                else:
                    wins_last_3_years = 0
            else:
                wins_last_3_years = 0
            
            base_career, base_last3 = _pre2018_wins(driver_num, year)
            career_wins += base_career
            wins_last_3_years += base_last3

            # 3. TrackType: 1 for street circuit, 0 for permanent
            track_type = is_street_circuit(track_name)
            
            # 4. ConstructorTrackAvg: Constructor's average finish at this specific track
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
                    fallback_year_data = historical_races[historical_races['Year'] == latest_year]
                    if not fallback_year_data.empty:
                        constructor_points = fallback_year_data.groupby('TeamName')['Points'].sum().sort_values(ascending=False)
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
            
            # 5. FormTrend: Momentum direction (improving vs declining)
            if round_num == 1:
                # First race: use previous season's trend
                prev_year_data = all_training_data[all_training_data['Year'] == year - 1]
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
            is_dnf = _is_dnf(status, position_text)
            
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
                'ActualGridPosition': actual_grid_position,  # Actual grid position from qualifying for this race
                'RecentForm': driver_recent_form,  # Last 5 races average finish (current momentum)
                'CareerWins': career_wins,  # Lifetime wins prior to this race (training set)
                'WinsLast3Years': wins_last_3_years,  # Wins in last 3 calendar years (recency / on fire)
                'PointsGapToLeader': points_gap,  # Points gap to championship leader
                'TrackType': track_type,  # 1 = street circuit, 0 = permanent
                'FormTrend': form_trend,  # Momentum direction (positive = improving)
                'DriverNumber': race.get('_RealNumber', race['DriverNumber']),
                'DriverName': race.get('Abbreviation', 'UNK'),
                'TeamName': race.get('TeamName', race.get('Team', '')),  # Constructor/team name
                'ActualPosition': race.get('Position', np.nan),
                'Points': race.get('TotalEventPoints', race.get('Points', 0)),  # Race + Sprint points (source of truth)
                'RacePoints': race.get('RacePoints', race.get('Points', 0)),  # Race points only (calculated from position)
                'SprintPoints': race.get('SprintPoints', 0),  # Sprint points only
                'IsDNF': is_dnf,  # Flag for DNF/DSQ/DNS
                'Status': status if pd.notna(status) else position_text if pd.notna(position_text) else ''
            }
            training_features.append(features)
    
    # Process test data similarly (no data leakage - use only data up to each race)
    if not test_data.empty:
        test_data_sorted = test_data.sort_values(['Year', 'RoundNumber'])
        
        for idx, race in test_data_sorted.iterrows():
            current_year = int(race['Year'])
            driver_num = str(race['DriverNumber'])
            track_name = race['EventName']
            round_num = race['RoundNumber']
            # Races in this season only (for multi-year test data)
            year_mask = test_data_sorted['Year'] == current_year
            races_this_season = test_data_sorted[year_mask]
            
            # Calculate features using only races BEFORE this one (same year)
            if round_num == 1:
                # Use previous year's data for first race (e.g. 2026 R1 uses 2025; 2025 R1 uses 2024)
                last_year = current_year - 1
                last_year_data = all_training_data[all_training_data['Year'] == last_year]
                if last_year_data.empty and not test_data.empty:
                    last_year_data = test_data[test_data['Year'] == last_year]
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
                # Use races from this test year up to (but not including) this race
                races_up_to_now = races_this_season[races_this_season['RoundNumber'] < round_num]
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
            
            # If no track-specific history, default to 10.0 for rookies (don't use overall average)
            if pd.isna(hist_track_avg):
                hist_track_avg = 10.0  # Default for rookies at this track
            
            # Store actual grid position from this race before calculating average
            actual_grid_position = race.get('GridPosition', np.nan)
            if pd.isna(actual_grid_position):
                # Try alternative column names
                actual_grid_position = race.get('StartingGrid', race.get('Grid', np.nan))
            
            # Get starting grid position - use AVERAGE grid position instead of actual
            # This matches what we'll use for future race predictions and eliminates train/test mismatch
            # For test data, we calculate average from training data + previous test races
            if round_num == 1:
                # First race: use only training data
                combined_data = all_training_data
            else:
                # Use training data + previous test races (this year only)
                previous_test_races = races_this_season[races_this_season['RoundNumber'] < round_num]
                combined_data = pd.concat([all_training_data, previous_test_races], ignore_index=True) if not previous_test_races.empty else all_training_data
            
            # Using season-specific average (only races from current season before current round)
            grid_position = calculate_average_grid_position(combined_data, driver_num, current_year, round_num, season_specific=True)
            
            # Fallback: if no historical data, use actual grid position from this race
            if pd.isna(grid_position):
                grid_position = actual_grid_position
            
            # Recent form (last 5 races average finish) - captures current momentum
            if round_num == 1:
                # First race: use previous year's recent form (last_year_data already set above)
                if not last_year_data.empty:
                    recent_form = calculate_recent_form(last_year_data, num_races=5)
                    driver_recent_form = recent_form.get(driver_num, np.nan)
                else:
                    driver_recent_form = np.nan
            else:
                # Use races from this test year up to (but not including) this race
                races_up_to_now = races_this_season[races_this_season['RoundNumber'] < round_num]
                if not races_up_to_now.empty:
                    recent_form = calculate_recent_form(races_up_to_now, num_races=5)
                    driver_recent_form = recent_form.get(driver_num, np.nan)
                else:
                    driver_recent_form = np.nan
            
            # Calculate new features for improved model
            # 1. PointsGapToLeader: Points gap to championship leader
            if round_num == 1:
                if not last_year_data.empty:
                    last_season_points = calculate_season_points(last_year_data)
                    max_points = max(last_season_points.values()) if last_season_points else 0
                    points_gap = max_points - driver_points if max_points > 0 else 0
                else:
                    points_gap = 0
            else:
                races_up_to_now = races_this_season[races_this_season['RoundNumber'] < round_num]
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
                combined_data, driver_constructor_standing, track_name, current_year, round_num
            )
            # Fallback: if no constructor track history, use constructor's overall average
            if pd.isna(constructor_track_avg):
                # Calculate constructor's overall average using TeamName if available
                historical_races = combined_data[
                    ((combined_data['Year'] < current_year) | 
                     ((combined_data['Year'] == current_year) & (combined_data['RoundNumber'] < round_num)))
                ].copy()
                
                if not historical_races.empty and 'TeamName' in historical_races.columns and 'Points' in historical_races.columns:
                    # Find teams with the target constructor standing
                    # Calculate standings for the most recent year available
                    latest_year = historical_races['Year'].max()
                    fallback_year_data = historical_races[historical_races['Year'] == latest_year]
                    if not fallback_year_data.empty:
                        constructor_points = fallback_year_data.groupby('TeamName')['Points'].sum().sort_values(ascending=False)
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
                # Use this test year's data only
                form_trend = calculate_form_trend(races_this_season, driver_num, round_num)
            
            # CareerWins and WinsLast3Years (must match training feature set for predict)
            prior_to_race = combined_data[
                (combined_data['Year'] < current_year) |
                ((combined_data['Year'] == current_year) & (combined_data['RoundNumber'] < round_num))
            ].copy()
            # One row per driver per race: canonicalize DriverNumber so 4/4.0/"4" don't double-count
            if not prior_to_race.empty and 'DriverNumber' in prior_to_race.columns:
                prior_to_race['_DNum'] = prior_to_race['DriverNumber'].apply(_canonical_driver_num)
                prior_to_race = prior_to_race.drop_duplicates(subset=['Year', 'RoundNumber', '_DNum'], keep='first')
                prior_to_race = prior_to_race.drop(columns=['_DNum'])
            driver_num_canon = _canonical_driver_num(driver_num)
            pos_col = 'ActualPosition' if 'ActualPosition' in prior_to_race.columns else 'Position'
            if pos_col in prior_to_race.columns:
                prior_to_race['_DNum'] = prior_to_race['DriverNumber'].apply(_canonical_driver_num)
                driver_prior = prior_to_race[prior_to_race['_DNum'] == driver_num_canon]
                driver_prior = driver_prior.drop_duplicates(subset=['Year', 'RoundNumber'], keep='first')
                career_wins = (driver_prior[pos_col] == 1).sum()
                prior_to_race = prior_to_race.drop(columns=['_DNum'], errors='ignore')
            else:
                career_wins = 0
            recent_start = max(combined_data['Year'].min(), current_year - 2)
            prior_3y = prior_to_race[prior_to_race['Year'] >= recent_start]
            if not prior_3y.empty and pos_col in prior_3y.columns:
                prior_3y = prior_3y.copy()
                prior_3y['_DNum'] = prior_3y['DriverNumber'].apply(_canonical_driver_num)
                driver_prior_3y = prior_3y[prior_3y['_DNum'] == driver_num_canon].drop_duplicates(subset=['Year', 'RoundNumber'], keep='first')
                wins_last_3_years = (driver_prior_3y[pos_col] == 1).sum()
            else:
                wins_last_3_years = 0
            base_career, base_last3 = _pre2018_wins(driver_num, current_year)
            career_wins += base_career
            wins_last_3_years += base_last3
            
            # Get DNF status if available
            status = race.get('Status', '')
            position_text = race.get('PositionText', '')
            is_dnf = _is_dnf(status, position_text)
            
            features = {
                'Year': current_year,
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
                'ActualGridPosition': actual_grid_position,  # Actual grid position from qualifying for this race
                'RecentForm': driver_recent_form,  # Last 5 races average finish (current momentum)
                'CareerWins': career_wins,  # Lifetime wins prior to this race
                'WinsLast3Years': wins_last_3_years,  # Wins in last 3 calendar years (recency / on fire)
                'PointsGapToLeader': points_gap,  # Points gap to championship leader
                'TrackType': track_type,  # 1 = street circuit, 0 = permanent
                'FormTrend': form_trend,  # Momentum direction (positive = improving)
                'DriverNumber': race.get('_RealNumber', race['DriverNumber']),
                'DriverName': race.get('Abbreviation', 'UNK'),
                'TeamName': race.get('TeamName', race.get('Team', '')),  # Constructor/team name
                'ActualPosition': race.get('Position', np.nan),
                'Points': race.get('TotalEventPoints', race.get('Points', 0)),  # Race + Sprint points (source of truth)
                'RacePoints': race.get('RacePoints', race.get('Points', 0)),  # Race points only (calculated from position)
                'SprintPoints': race.get('SprintPoints', 0),  # Sprint points only
                'IsDNF': is_dnf,  # Flag for DNF/DSQ/DNS
                'Status': status if pd.notna(status) else position_text if pd.notna(position_text) else ''
            }
            test_features.append(features)
    
    if reuse_training:
        training_df = pd.read_csv(Path('data') / 'training_data.csv')
    else:
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
        'features': ['SeasonPoints', 'SeasonStanding', 'SeasonAvgFinish',
                     'HistoricalTrackAvgPosition', 'ConstructorStanding', 'ConstructorTrackAvg',
                     'GridPosition', 'RecentForm', 'CareerWins', 'WinsLast3Years', 'TrackType'],
        'label': 'ActualPosition'
    }
    
    with open(Path(output_dir) / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)


def fetch_qualifying(year: int, rnd: int, event: str):
    """{driver abbreviation: qualifying position} for one race, or None if qualifying
    hasn't happened or isn't available.

    FastF1 looks events up by NAME and, when its calendar source is degraded, can
    silently hand back a different race (asking for Singapore once returned Hungary),
    so its answer is only used if the event name matches. Jolpica, asked by round
    number, is the fallback."""
    try:
        session = fastf1.get_session(year, event, 'Q')
        if str(session.event['EventName']) == event:
            session.load(laps=False, telemetry=False, weather=False, messages=False)
            res = session.results
            if res is not None and not res.empty and res['Position'].notna().any():
                return {str(a): int(p) for a, p in zip(res['Abbreviation'], res['Position']) if pd.notna(p)}
    except Exception:
        pass
    try:
        import urllib.request
        url = f'https://api.jolpi.ca/ergast/f1/{year}/{rnd}/qualifying.json'
        req = urllib.request.Request(url, headers={'User-Agent': 'formula-forecast'})
        with urllib.request.urlopen(req, timeout=30) as r:
            races = json.load(r)['MRData']['RaceTable']['Races']
        if races and event in races[0]['raceName'] and races[0]['QualifyingResults']:   # e.g. 'Bahrain Grand Prix in Malaysia'
            return {q['Driver']['code']: int(q['position']) for q in races[0]['QualifyingResults']}
    except Exception:
        pass
    return None


def save_next_qualifying(year: int):
    """Race results only get saved once a race has been run, so between qualifying
    and the race the qualifying result would otherwise be lost. Save it for the NEXT
    race to data/next_quali_<year>.json (used by rebuild/ to predict with the real
    grid); remove the file when there's nothing valid to save."""
    out = Path('data') / f'next_quali_{year}.json'
    schedule_path = Path('data') / f'schedule_{year}.json'
    upcoming = []
    if schedule_path.exists():
        done = set(pd.read_csv(Path('data') / 'test_data.csv').query('Year == @year')['RoundNumber'])
        upcoming = sorted((s['RoundNumber'], s['EventName'])
                          for s in json.loads(schedule_path.read_text(encoding='utf-8'))
                          if s['Year'] == year and s['RoundNumber'] not in done)
    positions = fetch_qualifying(year, *upcoming[0]) if upcoming else None
    if not positions:
        out.unlink(missing_ok=True)
        print('\nNext race qualifying: ' + (f'R{upcoming[0][0]} {upcoming[0][1]} not run yet'
                                           if upcoming else 'no upcoming race'))
        return
    rnd, event = upcoming[0]
    out.write_text(json.dumps({'Year': year, 'RoundNumber': rnd, 'EventName': event,
                               'positions': positions}, indent=1), encoding='utf-8')
    print(f'\nNext race qualifying saved: {year} R{rnd} {event} ({len(positions)} drivers) -> {out}')


def main():
    """Main function to collect and organize F1 data."""
    import sys
    force_refresh = '--refresh' in sys.argv
    force_reorganize = '--reorganize' in sys.argv

    # Training data: every completed season up to the last one
    training_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    # Test years (these are the ones that appear in the predict race selector)
    test_years = [2026]

    print("F1 Data Collection")
    print("=" * 50)
    print("Note: Collection is incremental - races already saved in data/raw/")
    print("      snapshots are loaded from disk; only NEW races are fetched.")
    print("      Use 'python collect_data.py --refresh' to force a full refetch.")
    print()
    print("Note: If you encounter API errors, the script will:")
    print("      - Retry failed requests up to 3 times")
    print("      - Skip problematic years and continue with available data")
    print("      - Continue even if some years fail to load")
    print()
    
    try:
        training_df, test_df = organize_data(training_years, test_years,
                                             force_refresh=force_refresh,
                                             force_reorganize=force_reorganize)
        if training_df is not None:
            save_data(training_df, test_df)

        # qualifying for the next race (after qualifying, before the race)
        save_next_qualifying(test_years[-1])

        if training_df is None:
            print("\nData collection complete (no changes - feature CSVs untouched).")
            return

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

