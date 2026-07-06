import React, { useState, useMemo } from 'react';
import { getCircuitImage } from '../circuitImages';

/**
 * Reusable race picker (year filter + search + circuit-art buttons).
 * Shared by the Predictions and Simulator tabs.
 *
 * Controlled inputs:
 *  - races, uniqueYears: from useRaces()
 *  - selectedRace, onSelect: current selection + handler
 *  - selectedYear, onYearChange: year filter
 *  - minimized: collapse to the sidebar layout once something is selected
 *  - onClear: optional clear button (shown only when minimized)
 *  - title: heading text
 *  - loading: disables buttons during fetches
 *  - showArt: render circuit background art (hidden when minimized)
 */
function RaceSelector({
  races,
  uniqueYears,
  selectedRace,
  onSelect,
  selectedYear,
  onYearChange,
  minimized = false,
  onClear,
  title,
  loading = false,
  showArt = true,
}) {
  const [searchQuery, setSearchQuery] = useState('');

  const filteredRaces = useMemo(() => {
    if (!Array.isArray(races)) return [];
    let filtered = races;
    if (selectedYear) {
      filtered = filtered.filter((race) => race.year === selectedYear);
    }
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(
        (race) => race.eventName && race.eventName.toLowerCase().includes(query)
      );
    }
    return filtered;
  }, [races, selectedYear, searchQuery]);

  return (
    <div className={`race-selector ${minimized ? 'minimized' : ''}`}>
      <div className="race-selector-header">
        <h2>{title || (minimized ? 'Races' : 'Select a Race')}</h2>
        {minimized && onClear && (
          <button className="clear-selection-btn" onClick={onClear} aria-label="Clear selection">
            &times;
          </button>
        )}
      </div>

      {uniqueYears.length > 0 && (
        <div className="year-filter">
          <label>Year:</label>
          <select
            value={selectedYear || ''}
            onChange={(e) => onYearChange(e.target.value ? parseInt(e.target.value) : null)}
          >
            <option value="">All</option>
            {uniqueYears.map((year) => (
              <option key={year} value={year}>
                {year}
              </option>
            ))}
          </select>
        </div>
      )}

      {!minimized && (
        <div className="search-filter">
          <input
            type="text"
            placeholder="Search races..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            aria-label="Search races"
          />
        </div>
      )}

      <div className="races-list">
        {loading && races.length === 0 ? (
          <div className="loading">Loading races...</div>
        ) : filteredRaces.length === 0 ? (
          <div className="no-races">No races found</div>
        ) : (
          filteredRaces.map((race, idx) => {
            const circuitImage = getCircuitImage(race.eventName);
            const isActive =
              selectedRace?.year === race.year &&
              selectedRace?.eventName === race.eventName &&
              selectedRace?.roundNumber === race.roundNumber;
            return (
              <button
                key={`${race.year}-${race.roundNumber}-${idx}`}
                className={`race-button ${isActive ? 'active' : ''}`}
                onClick={() => onSelect(race)}
                disabled={loading}
                style={circuitImage ? { '--circuit-image': `url(${circuitImage})` } : {}}
              >
                <div className="race-button-content">
                  <span className="race-name">{race.eventName}</span>
                  <span className="race-details">
                    {race.year} - Round {race.roundNumber}
                    {race.isFuture && <span className="future-badge">F</span>}
                  </span>
                </div>
                {circuitImage && showArt && !minimized && <div className="race-button-image"></div>}
              </button>
            );
          })
        )}
      </div>
    </div>
  );
}

export default RaceSelector;
