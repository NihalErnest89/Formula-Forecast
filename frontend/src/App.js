import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import './App.css';
import { getCircuitImage } from './circuitImages';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

function App() {
  const [races, setRaces] = useState([]);
  const [selectedRace, setSelectedRace] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedYear, setSelectedYear] = useState(null);
  const [filteredRaces, setFilteredRaces] = useState([]);
  const [showFiltered, setShowFiltered] = useState(false);

  const fetchRaces = useCallback(async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_BASE_URL}/api/races`);
      const sortedRaces = response.data.races.sort((a, b) => {
        if (a.year !== b.year) return b.year - a.year;
        return b.roundNumber - a.roundNumber;
      });
      setRaces(sortedRaces);
      setSelectedYear(prevYear => {
        if (!prevYear && sortedRaces.length > 0) {
          return sortedRaces[0].year;
        }
        return prevYear;
      });
    } catch (err) {
      setError('Failed to load races. Make sure the backend API is running.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchRaces();
  }, [fetchRaces]);

  useEffect(() => {
    if (selectedYear) {
      const filtered = races.filter(race => race.year === selectedYear);
      setFilteredRaces(filtered);
    } else {
      setFilteredRaces(races);
    }
  }, [selectedYear, races]);

  const handleRaceSelect = async (race) => {
    try {
      setLoading(true);
      setError(null);
      setSelectedRace(race);
      
      const response = await axios.post(`${API_BASE_URL}/api/predict`, {
        year: race.year,
        eventName: race.eventName,
        roundNumber: race.roundNumber
      });
      
      setPredictions(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to get predictions');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const uniqueYears = [...new Set(races.map(r => r.year))].sort((a, b) => b - a);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Formula Forecast</h1>
        <p>F1 Race Position Predictions</p>
      </header>

      <main className={`App-main ${predictions ? 'has-predictions' : ''}`}>
        {error && (
          <div className="error-message">
            {error}
          </div>
        )}

        <div className={`race-selector ${predictions ? 'minimized' : ''}`}>
          <h2>{predictions ? 'Races' : 'Select a Race'}</h2>
          
          {uniqueYears.length > 0 && (
            <div className="year-filter">
              <label>Year:</label>
              <select 
                value={selectedYear || ''} 
                onChange={(e) => setSelectedYear(parseInt(e.target.value))}
              >
                <option value="">All</option>
                {uniqueYears.map(year => (
                  <option key={year} value={year}>{year}</option>
                ))}
              </select>
            </div>
          )}

          <div className="races-list">
            {loading && races.length === 0 ? (
              <div className="loading">Loading races...</div>
            ) : filteredRaces.length === 0 ? (
              <div className="no-races">No races available</div>
            ) : (
              filteredRaces.map((race, idx) => {
                const circuitImage = getCircuitImage(race.eventName);
                // Debug: log if image not found
                if (!circuitImage && race.eventName) {
                  console.log(`No image found for: "${race.eventName}"`);
                }
                return (
                  <button
                    key={`${race.year}-${race.roundNumber}-${idx}`}
                    className={`race-button ${selectedRace?.year === race.year && 
                      selectedRace?.eventName === race.eventName && 
                      selectedRace?.roundNumber === race.roundNumber ? 'active' : ''}`}
                    onClick={() => handleRaceSelect(race)}
                    disabled={loading}
                    style={circuitImage ? {
                      '--circuit-image': `url(${circuitImage})`
                    } : {}}
                  >
                    <div className="race-button-content">
                      <span className="race-name">{race.eventName}</span>
                      <span className="race-details">
                        {race.year} - Round {race.roundNumber}
                        {race.isFuture && <span className="future-badge">F</span>}
                      </span>
                    </div>
                    {circuitImage && <div className="race-button-image"></div>}
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
                // Use sequential rank for display when filtered, original rank for error calculation
                const displayRank = showFiltered ? idx + 1 : pred.rank;
                // For filtered view, compare display rank to actual; for unfiltered, use original rank
                const error = hasActual ? Math.abs(displayRank - pred.actualPosition) : null;
                const errorClass = hasActual && !pred.isFiltered ? (
                  error === 0 ? 'exact' : 
                  error <= 1 ? 'close' : 
                  error <= 2 ? 'fair' : 'poor'
                ) : null;
                const isFiltered = pred.isFiltered || false;
                
                return (
                  <div 
                    key={idx} 
                    className={`prediction-row ${idx === 0 ? 'podium-gold' : idx === 1 ? 'podium-silver' : idx === 2 ? 'podium-bronze' : ''}`}
                  >
                    <div className="col-rank">
                      <span className={`rank-number ${idx === 0 ? 'gold' : idx === 1 ? 'silver' : idx === 2 ? 'bronze' : 'neutral'}`}>
                        {displayRank}
                      </span>
                    </div>
                    <div className="col-driver">
                      <span className="driver-name">{pred.driverName}</span>
                      {pred.constructor && (
                        <span className="constructor">{pred.constructor}</span>
                      )}
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
                                  Predicted Finish: {pred.rank}<br />
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
      </main>

      <footer className="App-footer">
        <p>Powered by deep neural networks trained on F1 data (2020-2024)</p>
      </footer>
    </div>
  );
}

export default App;

