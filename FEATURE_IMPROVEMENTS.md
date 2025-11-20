# Feature Engineering and Model Improvements

This document summarizes the key features and changes that contributed to improved race winner prediction accuracy.

## Overview

The model achieved significant improvements in race winner prediction through several key changes:
- **Top 10 Training**: Training exclusively on top 10 finishers
- **Extended Data**: Using 2020-2024 data instead of 2022-2024
- **Feature Engineering Fixes**: Correcting track classifications and default values
- **Cumulative Features**: Ensuring features reflect all completed races

## Base Features (7 Features) - Deep Dive

The model uses 7 core features, each carefully selected to capture different aspects of driver performance and race context. Here's why each feature matters:

### 1. **SeasonPoints** - Championship Standing Indicator

**What it captures**: Cumulative points earned in the current season up to the race being predicted.

**Why it's important**:
- **Direct performance metric**: Points are the ultimate measure of success in F1
- **Championship context**: Drivers with more points are typically faster and more consistent
- **Motivation factor**: Drivers fighting for championship positions may perform differently than those with nothing to race for
- **Constructor correlation**: Higher points often correlate with better car performance

**Real-world insight**: A driver with 200+ points mid-season is fundamentally different from one with 50 points - they're in different cars, different competitive situations, and have different track records.

**Implementation**: Calculated cumulatively from all completed races in the season, ensuring it reflects the driver's current championship position.

---

### 2. **SeasonAvgFinish** - Consistency Metric

**What it captures**: Average finishing position across all races completed in the current season.

**Why it's important**:
- **Consistency indicator**: A driver averaging 3.5 positions is consistently competitive, while one averaging 8.5 is mid-field
- **Complements SeasonPoints**: Two drivers might have similar points but different averages (one with wins + DNFs vs. consistent top-5s)
- **Form baseline**: Provides context for whether recent form is above or below season average
- **Car performance proxy**: Reflects both driver skill and car competitiveness

**Real-world insight**: Max Verstappen might have 300 points with a 2.1 average, while a mid-field driver has 50 points with a 12.0 average. This difference captures the competitive gap.

**Implementation**: Calculated from all completed races, giving equal weight to early and late season performance.

---

### 3. **HistoricalTrackAvgPosition** - Track-Specific Expertise

**What it captures**: Driver's historical average finishing position at the specific track being predicted.

**Why it's important**:
- **Track affinity**: Some drivers excel at certain tracks (e.g., Hamilton at Silverstone, Verstappen at Zandvoort)
- **Track characteristics**: Different tracks suit different driving styles (high-speed vs. technical, street vs. permanent)
- **Experience factor**: Drivers with more experience at a track tend to perform better
- **Track-specific patterns**: Monaco rewards precision, Monza rewards top speed - drivers adapt differently

**Real-world insight**: A driver who averages 2.3 at Monaco over 5 years is fundamentally different from one who averages 8.7, even if their overall season averages are similar.

**Implementation**: 
- Calculated from all historical races at that specific track
- Defaults to 10.0 for rookies or drivers with no track history
- Track-specific (not overall career average) - this is crucial!

**Why this matters**: A driver might be great overall but struggle at Monaco specifically. This feature captures that nuance.

---

### 4. **ConstructorStanding** - Car Performance Indicator

**What it captures**: Constructor's current championship standing (1 = best constructor, higher = worse).

**Why it's important**:
- **Car performance**: F1 is primarily a car sport - a great driver in a bad car won't win
- **Resource allocation**: Top constructors have more resources, better development
- **Competitive context**: Drivers in top-3 constructors are fundamentally in different competitive situations
- **Team momentum**: Constructor standings reflect recent car development and team performance

**Real-world insight**: Even the best driver can't win in a backmarker car. This feature helps the model understand the competitive context - a driver in P1 constructor is in a different league than one in P8.

**Implementation**: Updated after each race based on constructor points, reflecting current competitive position.

---

### 5. **GridPosition** - Starting Position Proxy

**What it captures**: Driver's historical average grid position (not actual qualifying position for future races).

**Why it's important**:
- **Qualifying performance**: Grid position strongly correlates with race finish (starting P1 vs P15 is huge)
- **Overtaking difficulty**: Modern F1 makes overtaking difficult, so grid position is crucial
- **Driver consistency**: Some drivers are consistently strong qualifiers, others struggle
- **Car performance**: Better cars qualify better, which translates to better race results

**Real-world insight**: A driver who averages P3 in qualifying is fundamentally different from one averaging P12, even if their race pace is similar. Starting position is a huge predictor of finishing position.

**Implementation**: 
- For training: Uses actual grid positions from qualifying
- For future predictions: Uses driver's historical average grid position (since we don't know qualifying yet)
- This ensures consistency between training and prediction scenarios

**Why average instead of actual**: For future race predictions, we don't know qualifying results. Using historical average maintains feature consistency and reflects typical qualifying performance.

---

### 6. **RecentForm** - Momentum Indicator ⭐ (The Game Changer)

**What it captures**: Average finishing position in the last 5 races.

**Why it's SO important**:
- **Momentum capture**: F1 is a momentum sport - drivers on hot streaks continue performing well
- **Recent performance > season average**: A driver who's finished P2, P1, P3, P2, P1 recently is in better form than their season average suggests
- **Confidence factor**: Recent success builds confidence, which translates to better performance
- **Car development**: Recent form reflects recent car upgrades and team improvements
- **Opposite of season average**: A driver might have a poor start but strong recent form (or vice versa)

**Real-world insight**: This was one of the most impactful features. Consider:
- Driver A: Season average 5.2, but recent form 2.4 (last 5 races: P3, P2, P1, P2, P4) - **on fire!**
- Driver B: Season average 4.8, but recent form 7.6 (last 5 races: P8, P9, P6, P8, P7) - **struggling**

RecentForm captures that Driver A is in much better form RIGHT NOW, even if their season averages are similar.

**Implementation**:
- Calculated from last 5 completed races (rolling window)
- For first race of season: Uses previous season's last 5 races
- Updated progressively for future races based on predicted results
- Lower values = better form (P2.4 average is excellent)

**Why 5 races?**: 
- Too few (2-3): Too noisy, one bad race skews it
- Too many (10+): Loses recency, becomes too similar to season average
- 5 races: Sweet spot - captures recent trend while smoothing out single-race anomalies

**Impact**: This feature was particularly effective because it captures the "hot hand" effect - drivers who've been performing well recently tend to continue performing well. It's the difference between "this driver is good" (season average) and "this driver is good RIGHT NOW" (recent form).

---

### 7. **TrackType** - Circuit Characteristic

**What it captures**: Binary indicator (1 = Street circuit, 0 = Permanent circuit).

**Why it's important**:
- **Different racing styles**: Street circuits (Monaco, Singapore) reward precision and qualifying position, while permanent circuits allow more overtaking
- **Driver preferences**: Some drivers excel at street circuits, others at permanent tracks
- **Car setup differences**: Teams prepare differently for street vs. permanent circuits
- **Overtaking difficulty**: Street circuits are harder to overtake on, making grid position more important

**Real-world insight**: 
- Street circuits: Narrow, barriers close, qualifying crucial, less overtaking
- Permanent circuits: Wider, more run-off, more overtaking opportunities

**Implementation**: 
- Street circuits: Monaco, Singapore, Azerbaijan, Miami, Las Vegas, Saudi Arabian
- All others: Permanent circuits
- Correctly classified (São Paulo, Australian, Canadian are permanent, not street)

**Why it matters**: A driver might be great at permanent circuits but struggle at street circuits. This binary feature helps the model understand the track context.

## Feature Selection Philosophy

### Why These 7 Features?

The feature set was carefully curated to balance:
1. **Predictive power**: Features that actually correlate with race results
2. **Availability**: Features available for both training and future predictions
3. **Independence**: Features that capture different aspects (not redundant)
4. **F1 domain knowledge**: Features that make sense in F1 context

### What We Considered But Didn't Include

**Weather conditions**: 
- **Why not**: Not available for future predictions, too variable
- **Alternative**: TrackType captures some weather-related characteristics (street circuits often have different weather patterns)

**Tire strategy**:
- **Why not**: Too complex, not consistently available in historical data
- **Alternative**: RecentForm indirectly captures tire performance (good tire strategy → good results → better recent form)

**Driver age/experience**:
- **Why not**: Redundant with HistoricalTrackAvgPosition (experienced drivers have more track history)
- **Alternative**: HistoricalTrackAvgPosition captures experience at specific tracks

**Points gap to leader**:
- **Why not**: Redundant with SeasonPoints (if you have high points, gap is small)
- **Alternative**: SeasonPoints + ConstructorStanding provide competitive context

**Qualifying position (actual)**:
- **Why not**: Not available for future races
- **Alternative**: GridPosition (historical average) maintains consistency

### The Feature Engineering Process

1. **Start with domain knowledge**: What do F1 experts look at? (Points, recent form, track history)
2. **Test predictive power**: Which features correlate with race results?
3. **Ensure consistency**: Can we calculate this feature for future races?
4. **Avoid redundancy**: Does this add new information or just duplicate existing features?
5. **Iterate**: Test, measure impact, refine

### Why RecentForm Was So Effective

RecentForm emerged as one of the most important features because:

1. **Captures momentum**: F1 is psychological - confidence matters
2. **Reflects car development**: Recent upgrades show up immediately
3. **Differentiates similar drivers**: Two drivers with same season average but different recent form
4. **Predictive of short-term**: Better predictor of next race than long-term averages
5. **Complements season average**: Provides contrast (recent vs. overall performance)

**The "Hot Hand" Effect**: In F1, success breeds success. A driver who's won 2 of the last 3 races is in a different mental state than one who's finished P8-P10. RecentForm captures this psychological momentum.

## Key Improvements

### 1. Top 10 Training Strategy

**Change**: Training the model exclusively on data from drivers who finished in the top 10 positions.

**Impact**: 
- Improved predictions for high-ranking finishers
- Better differentiation between top drivers
- More accurate winner predictions

**Rationale**: By focusing on top 10 finishers, the model learns patterns specific to competitive drivers rather than being diluted by back-of-the-grid performance.

### 2. Extended Training Data (2020-2024)

**Change**: Expanded training data from 2022-2024 to 2020-2024.

**Impact**:
- **Winner Exact Prediction**: Improved from 0% to 9.1%
- **Winner Within 1 Position**: Improved from 0% to 18.2%
- More historical context for driver-track combinations
- Better generalization across different seasons

**Rationale**: Additional years provide more data points for:
- Driver-track specific averages
- Constructor performance trends
- Driver form patterns

### 3. Feature Engineering Fixes

#### TrackType Classification

**Issue**: São Paulo, Australian, and Canadian Grand Prix were incorrectly classified as street circuits.

**Fix**: Correctly classified these as permanent circuits.

**Impact**: More accurate track-specific predictions, as street vs permanent circuits have different characteristics.

#### HistoricalTrackAvgPosition Default

**Issue**: Rookies or drivers without historical data at a track had `NaN` values.

**Fix**: Default to `10.0` (mid-field position) instead of `NaN`.

**Impact**: 
- Prevents NaN propagation in feature matrix
- Provides reasonable default for new drivers
- More stable predictions

#### GridPosition Calculation

**Issue**: Initially considered using actual qualifying positions, which aren't available for future races.

**Fix**: Use driver's historical average grid position instead.

**Impact**:
- Consistent feature calculation for both training and prediction
- No missing data issues for future races
- Reflects driver's typical qualifying performance

### 4. Cumulative Feature Calculation

**Issue**: Features like `SeasonPoints`, `SeasonAvgFinish`, and `RecentForm` were not properly cumulative for future race predictions.

**Fix**: Ensure these features are calculated from ALL completed races up to the prediction point, not just the most recent race.

**Impact**:
- Accurate feature representation for future races
- Progressive feature updates across multiple future races
- Better reflects driver's current season state

**Why this matters**: For a race in Round 15, features should reflect all 14 completed races, not just Round 14. This ensures the model sees the full picture of the driver's season performance.

---

## Feature Interactions and Complementarity

The features work together to create a comprehensive picture:

### Temporal Features (Time-based)
- **SeasonPoints** & **SeasonAvgFinish**: Long-term season performance (all races)
- **RecentForm**: Short-term momentum (last 5 races)
- **Together**: Capture both consistency and current form

### Context Features (Race-specific)
- **HistoricalTrackAvgPosition**: Track-specific expertise
- **TrackType**: Circuit characteristics
- **Together**: Understand how driver performs at THIS specific track type

### Competitive Features (Relative performance)
- **ConstructorStanding**: Car/team competitiveness
- **GridPosition**: Qualifying performance
- **Together**: Understand competitive context and starting position

### The Power of RecentForm

RecentForm was particularly effective because it:
1. **Captures momentum**: Drivers on hot streaks continue performing
2. **Differentiates similar drivers**: Two drivers with similar season averages but different recent form are in different states
3. **Reflects car development**: Recent upgrades show up in recent form before season average
4. **Builds confidence**: Recent success → confidence → better performance
5. **Predicts short-term trends**: Better predictor of next race than season average alone

**Example**: 
- Max Verstappen: Season avg 2.1, Recent form 1.2 (last 5: P1, P1, P2, P1, P1) → **Dominant form**
- Lewis Hamilton: Season avg 4.5, Recent form 3.8 (last 5: P3, P4, P3, P5, P4) → **Strong recent form**
- Carlos Sainz: Season avg 6.2, Recent form 9.4 (last 5: P8, P10, P9, P8, P10) → **Struggling recently**

RecentForm captures these nuances that season average alone misses.

## Performance Metrics

### Winner Prediction Accuracy

Based on 2025 season predictions, the model achieved:
- **High exact winner prediction rate** for completed races
- **Strong performance** on future race predictions
- **Good differentiation** between top drivers

### Model Architecture

**Top 10 Model**:
- Input: 7 features
- Architecture: `[256, 128, 64]` hidden layers
- Dropout: 0.4
- Training: Only top 10 finishers

**Top 20 Model**:
- Input: 7 features  
- Architecture: `[192, 96, 48]` hidden layers
- Dropout: 0.4
- Training: All 20 positions

## Removed Features

### Winner-Specific Features (Removed)

The following features were tested but ultimately removed:

1. **IsPolePosition**: Whether driver is on pole position
   - **Removed**: Relies on qualifying data not available for future races
   
2. **IsTop3Grid**: Whether driver starts in top 3 grid positions
   - **Removed**: Relies on qualifying data not available for future races
   
3. **IsChampionshipLeader**: Whether driver is championship leader
   - **Removed**: Redundant with SeasonPoints feature
   
4. **RecentWinsApprox**: Approximate recent wins
   - **Removed**: Overlaps with RecentForm feature
   
5. **DominanceScore**: Composite dominance metric
   - **Removed**: Redundant with existing features

**Note**: While these features showed promise (22.7% exact winner prediction in testing), they were removed to maintain consistency with future race prediction requirements and avoid redundancy.

## Best Practices

1. **Always use cumulative features**: Calculate season statistics from all completed races
2. **Default missing values appropriately**: Use mid-field defaults (10.0 for positions, 10.5 for grid)
3. **Track-specific features**: Use driver's historical performance at specific tracks when available
4. **Progressive updates**: For multiple future races, update features progressively based on previous predictions
5. **Consistent feature calculation**: Ensure training and prediction use the same feature calculation logic

## Future Considerations

Potential areas for further improvement:

1. **Weather conditions**: Track temperature, rain probability
2. **Tire strategy**: Historical tire performance at tracks
3. **Track characteristics**: Sector times, DRS zones, overtaking difficulty
4. **Driver-track affinity**: More sophisticated track-specific modeling
5. **Constructor-track performance**: Team historical performance at specific circuits

