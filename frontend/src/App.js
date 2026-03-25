import React, { useState, useEffect, useCallback } from 'react';
import './App.css';
import { getCircuitImage } from './circuitImages';
import { getTeamColor, getDriverImage } from './driverData';

const DATA_BASE = process.env.PUBLIC_URL + '/data';

function App() {
  const [races, setRaces] = useState([]);
  const [filteredRaces, setFilteredRaces] = useState([]);
  const [selectedYear, setSelectedYear] = useState(2025);
  const [selectedRace, setSelectedRace] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showFiltered, setShowFiltered] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  const fetchRaces = useCallback(async () => {
    try {
      const response = await fetch(`${DATA_BASE}/races.json`);
      const json = await response.json();
      const racesData = json?.races || (Array.isArray(json) ? json : []);
      setRaces(racesData);
    } catch (err) {
      console.error('Error fetching races:', err);
      setError('Failed to load races');
      setRaces([]);
    }
  }, []);

  useEffect(() => {
    fetchRaces();
  }, [fetchRaces]);

  useEffect(() => {
    // Ensure races is always an array
    if (!Array.isArray(races)) {
      setFilteredRaces([]);
      return;
    }
    
    let filtered = races;
    
    if (selectedYear) {
      filtered = filtered.filter(race => race.year === selectedYear);
    }
    
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(race => 
        race.eventName && race.eventName.toLowerCase().includes(query)
      );
    }
    
    setFilteredRaces(filtered);
  }, [selectedYear, races, searchQuery]);

  const handleRaceSelect = async (race) => {
    try {
      setLoading(true);
      setError(null);
      setSelectedRace(race);

      const response = await fetch(`${DATA_BASE}/predictions/${race.year}-${race.roundNumber}.json`);
      if (!response.ok) throw new Error('Predictions not found');
      const data = await response.json();
      setPredictions(data);
    } catch (err) {
      setError('Failed to get predictions');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleClearSelection = () => {
    setSelectedRace(null);
    setPredictions(null);
    setShowFiltered(false);
  };

  const uniqueYears = Array.isArray(races) 
    ? [...new Set(races.map(r => r.year).filter(Boolean))].sort((a, b) => b - a)
    : [];

  return (
    <div className="App">
      <header className="App-header">
        <h1>Formula Forecast</h1>
        <p>F1 Race Position Predictions</p>
      </header>

      <main className={`App-main ${predictions ? 'has-predictions' : ''}`}>
        <>
            {error && (
              <div className="error-message">{error}</div>
            )}

            <div className={`race-selector ${predictions ? 'minimized' : ''}`}>
              <div className="race-selector-header">
                <h2>{predictions ? 'Races' : 'Select a Race'}</h2>
                {predictions && (
                  <button className="clear-selection-btn" onClick={handleClearSelection}>
                    &times;
                  </button>
                )}
              </div>

              {uniqueYears.length > 0 && (
                <div className="year-filter">
                  <label>Year:</label>
                  <select
                    value={selectedYear || ''}
                    onChange={(e) => setSelectedYear(e.target.value ? parseInt(e.target.value) : null)}
                  >
                    <option value="">All</option>
                    {uniqueYears.map(year => (
                      <option key={year} value={year}>{year}</option>
                    ))}
                  </select>
                </div>
              )}

              {!predictions && (
                <div className="search-filter">
                  <input
                    type="text"
                    placeholder="Search races..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
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
                    return (
                      <button
                        key={`${race.year}-${race.roundNumber}-${idx}`}
                        className={`race-button ${selectedRace?.year === race.year &&
                          selectedRace?.eventName === race.eventName &&
                          selectedRace?.roundNumber === race.roundNumber ? 'active' : ''}`}
                        onClick={() => handleRaceSelect(race)}
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
                        {circuitImage && !predictions && <div className="race-button-image"></div>}
                      </button>
                    );
                  })
                )}
              </div>
            </div>

            {loading && predictions === null && selectedRace && (
              <div className="loading-predictions">
                <div className="spinner"></div>
                <p>Calculating predictions...</p>
              </div>
            )}

            {predictions && !loading && (
              <div className="predictions-container">
                <div className="race-info">
                  <h2>{predictions.race.eventName}</h2>
                  <p>
                    {predictions.race.year} - Round {predictions.race.roundNumber}
                    {predictions.race.isFuture && <span className="future-badge">Future Race</span>}
                  </p>

                  {!predictions.race.isFuture && (
                    <div className="filter-toggle">
                      <label className="filter-checkbox-label tooltip-trigger">
                        <input
                          type="checkbox"
                          checked={showFiltered}
                          onChange={(e) => setShowFiltered(e.target.checked)}
                          className="filter-checkbox"
                        />
                        <span>Show filtered rankings</span>
                        <span className="tooltip tooltip-left">
                          <span className="tooltip-content">
                            Excludes DNFs/DSQ and more than 6 place drops from start.
                          </span>
                        </span>
                      </label>
                    </div>
                  )}
                </div>

                <div className="predictions-table">
                  <div className="table-header">
                    <div className="col-rank">Rank</div>
                    <div className="col-driver">Driver</div>
                    <div className="col-actual">Actual</div>
                  </div>

                  {(showFiltered ? predictions.predictionsFiltered : predictions.predictionsUnfiltered).map((pred, idx) => {
                    const hasActual = pred.actualPosition !== null && pred.actualPosition !== undefined;
                    const displayRank = showFiltered ? idx + 1 : pred.rank;
                    const error = hasActual ? Math.abs(displayRank - pred.actualPosition) : null;
                    const errorClass = hasActual && !pred.isFiltered ? (
                      error === 0 ? 'exact' :
                      error <= 1 ? 'close' :
                      error <= 2 ? 'fair' : 'poor'
                    ) : null;
                    const isFiltered = pred.isFiltered || false;
                    const teamColor = getTeamColor(pred.constructor);

                    return (
                      <div
                        key={idx}
                        className={`prediction-row ${idx === 0 ? 'podium-gold' : idx === 1 ? 'podium-silver' : idx === 2 ? 'podium-bronze' : ''}`}
                        style={{ animationDelay: `${idx * 0.05}s` }}
                      >
                        <div className="col-rank">
                          <span className={`rank-number ${idx === 0 ? 'gold' : idx === 1 ? 'silver' : idx === 2 ? 'bronze' : 'neutral'}`}>
                            {displayRank}
                          </span>
                        </div>
                        <div
                          className="col-driver driver-row-with-image"
                          style={{
                            '--team-color': teamColor.primary,
                            '--team-color-secondary': teamColor.secondary
                          }}
                        >
                          <span className="driver-name">{pred.driverName}</span>
                          {pred.constructor && (
                            <span className="constructor">{pred.constructor}</span>
                          )}
                          <div className="driver-row-image">
                            {(() => {
                              const driverImg = getDriverImage(pred.driverName);
                              console.log('Driver:', pred.driverName, 'Image URL:', driverImg);
                              if (driverImg) {
                                return (
                                  <>
                                    <img
                                      src={driverImg}
                                      alt={pred.driverName}
                                      className="driver-face-image"
                                      onLoad={() => console.log('Image loaded:', pred.driverName)}
                                      onError={(e) => {
                                        console.error('Image failed to load:', driverImg, 'for driver:', pred.driverName);
                                        e.target.style.display = 'none';
                                        const fallback = e.target.nextElementSibling;
                                        if (fallback) fallback.style.display = 'flex';
                                      }}
                                    />
                                    <div
                                      className="driver-row-number"
                                      data-driver={pred.driverName}
                                      style={{ display: 'none' }}
                                    >
                                      {pred.driverNumber || '?'}
                                    </div>
                                  </>
                                );
                              }
                              console.log('No image found for:', pred.driverName);
                              return (
                                <div className="driver-row-number">
                                  {pred.driverNumber || '?'}
                                </div>
                              );
                            })()}
                          </div>
                        </div>
                        <div className={`col-actual ${errorClass ? `accuracy-${errorClass}` : ''} ${isFiltered ? 'filtered' : ''}`}>
                          {hasActual ? (
                            <span className={`actual-value tooltip-trigger ${isFiltered ? 'filtered' : ''}`}>
                              {pred.actualPosition}
                              <span className="tooltip">
                                <span className="tooltip-title">
                                  {isFiltered ? 'Filtered Result' : 'Prediction Accuracy'}
                                </span>
                                <span className="tooltip-content">
                                  {isFiltered ? (
                                    <>
                                      Filtered: {pred.filterReason}<br />
                                      Start: {pred.gridPosition !== null ? pred.gridPosition : 'N/A'}<br />
                                      Predicted Finish: {displayRank}<br />
                                      Actual Finish: {pred.actualPosition}<br />
                                      Error: {error} position{error !== 1 ? 's' : ''}
                                    </>
                                  ) : (
                                    <>
                                      Start: {pred.gridPosition !== null ? pred.gridPosition : 'N/A'}<br />
                                      Predicted Finish: {displayRank}<br />
                                      Actual Finish: {pred.actualPosition}<br />
                                      Error: {error} position{error !== 1 ? 's' : ''}
                                    </>
                                  )}
                                </span>
                              </span>
                            </span>
                          ) : (
                            <span className="actual-unknown tooltip-trigger">
                              —
                              <span className="tooltip">
                                <span className="tooltip-content">No actual result available</span>
                              </span>
                            </span>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>

                <div className="predictions-footer">
                  <p>Total drivers analyzed: {predictions.totalDrivers}</p>
                  {!predictions.race.isFuture && (
                    <p className="filter-info">
                      {showFiltered
                        ? `Showing ${predictions.predictionsFiltered.length} filtered drivers (excludes DNFs and position drops > 6 places)`
                        : `Showing ${predictions.predictionsUnfiltered.length} drivers (includes all results)`}
                    </p>
                  )}
                </div>
              </div>
            )}
        </>
      </main>

      <footer className="App-footer">
        <p>Powered by deep neural networks trained on FastF1 data (2020-2024)</p>
        <p>Created by Nihal Ernest</p>
      </footer>
    </div>
  );
}

export default App;
