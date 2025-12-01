import pandas as pd
from pathlib import Path
from feature_calculation import calculate_future_race_features

def get_future_races(year: int):
    """
    Get future races (scheduled but not yet completed) from Fast F1 schedule.
    Uses cached schedule data without loading race sessions to avoid API calls.
    
    Args:
        year: Season year
        
    Returns:
        DataFrame with future race schedule information
    """
    try:
        # Lazy import to avoid triggering cache on every predict.py run
        import fastf1
        
        # Only get schedule - don't load sessions (that triggers API calls)
        schedule = fastf1.get_event_schedule(year)
        if schedule is None or schedule.empty:
            return pd.DataFrame()
        
        # Get all races from schedule
        # We'll determine if they're future races by checking if they exist in test data
        # This avoids loading sessions which triggers API calls
        all_races = []
        for _, event in schedule.iterrows():
            # Skip Pre-Season Testing (not a real race)
            event_name = event.get('EventName', '')
            if 'Pre-Season' in event_name or 'Pre Season' in event_name or 'Testing' in event_name:
                continue
            
            all_races.append({
                'Year': year,
                'EventName': event['EventName'],
                'RoundNumber': event['RoundNumber'],
                'Date': event.get('EventDate', ''),
                'Location': event.get('Location', '')
            })
        
        return pd.DataFrame(all_races)
    except Exception as e:
        # If fastf1 isn't available or fails, just return empty
        return pd.DataFrame()


def select_race_interactive(test_df: pd.DataFrame, training_df: pd.DataFrame = None):
    """
    Interactive function to let user select year and race from available options.
    Includes future races that haven't happened yet.
    
    Args:
        test_df: DataFrame with test data containing Year, EventName, RoundNumber columns
        training_df: Optional DataFrame with training data for historical track averages
        
    Returns:
        Tuple of (selected_df, input_source_string, is_future_race) or (None, None, False) if cancelled.
        If user types "all" for 2025, returns (list_of_race_tuples, "All 2025 races", 'all')
        where list_of_race_tuples is a list of (race_df, input_source, is_future) tuples.
    """
    if 'Year' not in test_df.columns or 'EventName' not in test_df.columns:
        return None, None, False
    
    # Get unique years from test data
    unique_years = sorted(test_df['Year'].unique())
    
    # Note: Future race checking is done lazily when user selects a year
    # to avoid triggering Fast F1 API calls on every predict.py run
    
    print("\n" + "=" * 70)
    print("Available Years:")
    print("=" * 70)
    for idx, year in enumerate(unique_years, 1):
        print(f"  {idx}. {year}")
    
    while True:
        try:
            year_choice = input(f"\nSelect year (1-{len(unique_years)}) or 'q' to quit: ").strip()
            if year_choice.lower() == 'q':
                return None, None, False
            
            year_idx = int(year_choice) - 1
            if 0 <= year_idx < len(unique_years):
                selected_year = unique_years[year_idx]
                break
            else:
                print(f"Please enter a number between 1 and {len(unique_years)}")
        except ValueError:
            print("Please enter a valid number or 'q' to quit")
    
    # Get races from test data (completed races)
    year_data = test_df[test_df['Year'] == selected_year]
    completed_races = year_data[['Year', 'EventName', 'RoundNumber']].drop_duplicates()
    
    # Get future races for this year
    future_races = get_future_races(selected_year)
    
    # Combine completed and future races
    if not future_races.empty:
        # Merge future races (avoid duplicates)
        future_races_clean = future_races[['Year', 'EventName', 'RoundNumber']].drop_duplicates()
        # Only add future races that aren't already in completed races
        future_only = future_races_clean[
            ~future_races_clean.set_index(['Year', 'EventName', 'RoundNumber']).index.isin(
                completed_races.set_index(['Year', 'EventName', 'RoundNumber']).index
            )
        ]
        all_races = pd.concat([completed_races, future_only], ignore_index=True)
    else:
        all_races = completed_races
    
    all_races = all_races.sort_values('RoundNumber')
    
    # Check which races are future (no data in test_df)
    is_future_list = []
    for _, race in all_races.iterrows():
        has_data = not test_df[
            (test_df['Year'] == race['Year']) & 
            (test_df['EventName'] == race['EventName']) &
            (test_df['RoundNumber'] == race['RoundNumber'])
        ].empty
        is_future_list.append(not has_data)
    
    print(f"\n" + "=" * 70)
    print(f"Available Races for {selected_year}:")
    print("=" * 70)
    for idx, (_, race) in enumerate(all_races.iterrows(), 1):
        future_marker = " [FUTURE]" if is_future_list[idx-1] else ""
        print(f"  {idx}. {race['EventName']} (Round {race['RoundNumber']}){future_marker}")
    
    while True:
        try:
            race_choice = input(f"\nSelect race (1-{len(all_races)}) or 'q' to quit" + (f" or 'all' to predict all {selected_year} races" if selected_year == 2025 else "") + ": ").strip()
            if race_choice.lower() == 'q':
                return None, None, False
            
            # Check for "all" option (only for 2025)
            if race_choice.lower() == 'all' and selected_year == 2025:
                # Process all races for 2025 and return list of race dataframes
                all_race_data = []
                print(f"\nProcessing all {len(all_races)} races for {selected_year}...")
                for idx, (_, race_row) in enumerate(all_races.iterrows(), 1):
                    race_name = race_row['EventName']
                    race_round = race_row['RoundNumber']
                    print(f"  [{idx}/{len(all_races)}] Processing {race_name} (Round {race_round})...")
                    
                    # Check if this is a future race (no data in test_df)
                    race_df = test_df[
                        (test_df['Year'] == race_row['Year']) & 
                        (test_df['EventName'] == race_row['EventName']) &
                        (test_df['RoundNumber'] == race_row['RoundNumber'])
                    ].copy()
                    
                    if race_df.empty:
                        # Future race - calculate features using most recent race data
                        try:
                            race_df = calculate_future_race_features(
                                test_df, selected_year, race_round, race_name, 
                                training_df if training_df is not None else None
                            )
                            input_source = f"{race_name} ({selected_year}, Round {race_round}) [FUTURE]"
                            all_race_data.append((race_df, input_source, True))
                        except Exception as e:
                            print(f"    Error calculating features for {race_name}: {e}")
                            print(f"    Skipping this race.")
                    else:
                        input_source = f"{race_name} ({selected_year}, Round {race_round})"
                        all_race_data.append((race_df, input_source, False))
                
                # Return special marker to indicate "all races" mode
                return all_race_data, f"All {selected_year} races", 'all'
            
            race_idx = int(race_choice) - 1
            if 0 <= race_idx < len(all_races):
                selected_race = all_races.iloc[race_idx]
                break
            else:
                print(f"Please enter a number between 1 and {len(all_races)}")
        except ValueError:
            if race_choice.lower() == 'all' and selected_year != 2025:
                print("'all' option is only available for 2025. Please select a specific race.")
            else:
                print("Please enter a valid number or 'q' to quit")
    
    # Check if this is a future race (no data in test_df)
    race_df = test_df[
        (test_df['Year'] == selected_race['Year']) & 
        (test_df['EventName'] == selected_race['EventName']) &
        (test_df['RoundNumber'] == selected_race['RoundNumber'])
    ].copy()
    
    if race_df.empty:
        # Future race - calculate features using most recent race data
        print(f"\n  This is a future race. Calculating features from most recent race data...")
        try:
            # Calculate features for future race using most recent completed race
            # Load training data if not already loaded
            if 'training_df' not in locals():
                script_dir = Path(__file__).parent
                training_data_path = script_dir.parent / 'data' / 'training_data.csv'
                training_df_local = None
                if training_data_path.exists():
                    try:
                        training_df_local = pd.read_csv(training_data_path)
                        # Filter out Pre-Season Testing (not a real race)
                        if 'EventName' in training_df_local.columns:
                            training_df_local = training_df_local[~training_df_local['EventName'].str.contains('Pre-Season|Pre Season|Testing', case=False, na=False)].copy()
                    except Exception:
                        pass
            else:
                training_df_local = training_df
            
            race_df = calculate_future_race_features(
                test_df, selected_year, selected_race['RoundNumber'], selected_race['EventName'], training_df_local
            )
            print(f"  Calculated features for {len(race_df)} drivers")
            print(f"  Note: AvgGridPosition is unknown for future races (will use historical average)")
            
            input_source = f"{selected_race['EventName']} ({selected_year}, Round {selected_race['RoundNumber']}) [FUTURE]"
            return race_df, input_source, True
        except Exception as e:
            print(f"  Error calculating features for future race: {e}")
            print(f"  Please use --input-file with driver features for future races.")
            return None, None, False
    
    input_source = f"{selected_race['EventName']} ({selected_year}, Round {selected_race['RoundNumber']})"
    
    return race_df, input_source, False

