import React, { useState } from 'react';
import { getTeamColorForDriver, getDriverImage } from '../driverData';
import { DATA_BASE } from '../hooks/useRaces';
import RaceSelector from './RaceSelector';

function PredictionsView({ races, uniqueYears }) {
  const [selectedYear, setSelectedYear] = useState(2025);
  const [selectedRace, setSelectedRace] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showFiltered, setShowFiltered] = useState(false);

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

  return (
    <div className={`view-body ${predictions ? 'has-predictions' : ''}`}>
      {error && <div className="error-message">{error}</div>}

      <RaceSelector
        races={races}
        uniqueYears={uniqueYears}
        selectedRace={selectedRace}
        onSelect={handleRaceSelect}
        selectedYear={selectedYear}
        onYearChange={setSelectedYear}
        minimized={!!predictions}
        onClear={handleClearSelection}
        loading={loading}
      />

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
              <div className="col-gap tooltip-trigger">
                Gap
                <span className="tooltip tooltip-left">
                  <span className="tooltip-content">
                    Predicted pace gap behind the leader, like a timing tower: short bars =
                    close fight, long bars = cleared off. Measured in predicted positions.
                  </span>
                </span>
              </div>
              <div className="col-actual">Actual</div>
            </div>

            {(() => {
              const list = showFiltered
                ? predictions.predictionsFiltered
                : predictions.predictionsUnfiltered;
              const scores = list.map((p) => p.predictedPosition);
              const leaderScore = scores.length ? scores[0] : 0;
              const maxGap = scores.length ? Math.max(...scores.map((s) => s - leaderScore)) : 0;
              return list.map((pred, idx) => {
                const gapToLeader =
                  pred.predictedPosition != null ? pred.predictedPosition - leaderScore : null;
                const interval =
                  idx === 0
                    ? null
                    : pred.predictedPosition != null && list[idx - 1].predictedPosition != null
                    ? pred.predictedPosition - list[idx - 1].predictedPosition
                    : null;
                const barPct =
                  gapToLeader != null && maxGap > 0 ? (gapToLeader / maxGap) * 100 : 0;
                pred = { ...pred, _gapToLeader: gapToLeader, _interval: interval, _barPct: barPct };
                const hasActual = pred.actualPosition !== null && pred.actualPosition !== undefined;
                const displayRank = showFiltered ? idx + 1 : pred.rank;
                const posError = hasActual ? Math.abs(displayRank - pred.actualPosition) : null;
                const errorClass =
                  hasActual && !pred.isFiltered
                    ? posError === 0
                      ? 'exact'
                      : posError <= 1
                      ? 'close'
                      : posError <= 2
                      ? 'fair'
                      : 'poor'
                    : null;
                const isFiltered = pred.isFiltered || false;
                const teamColor = getTeamColorForDriver(pred.driverName);

                return (
                  <div
                    key={idx}
                    className={`prediction-row ${
                      idx === 0
                        ? 'podium-gold'
                        : idx === 1
                        ? 'podium-silver'
                        : idx === 2
                        ? 'podium-bronze'
                        : ''
                    }`}
                    style={{ animationDelay: `${idx * 0.05}s` }}
                  >
                    <div className="col-rank">
                      <span
                        className={`rank-number ${
                          idx === 0 ? 'gold' : idx === 1 ? 'silver' : idx === 2 ? 'bronze' : 'neutral'
                        }`}
                      >
                        {displayRank}
                      </span>
                    </div>
                    <div
                      className="col-driver driver-row-with-image"
                      style={{
                        '--team-color': teamColor.primary,
                        '--team-color-secondary': teamColor.secondary,
                      }}
                    >
                      <span className="driver-name">{pred.driverName}</span>
                      {pred.constructor && <span className="constructor">{pred.constructor}</span>}
                      <div className="driver-row-image">
                        {(() => {
                          const driverImg = getDriverImage(pred.driverName);
                          if (driverImg) {
                            return (
                              <>
                                <img
                                  src={driverImg}
                                  alt={pred.driverName}
                                  className="driver-face-image"
                                  onError={(e) => {
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
                          return <div className="driver-row-number">{pred.driverNumber || '?'}</div>;
                        })()}
                      </div>
                    </div>
                    <div className="col-gap">
                      {pred._gapToLeader != null ? (
                        idx === 0 ? (
                          <span className="gap-label leader">Leader</span>
                        ) : (
                          <>
                            <div className="gap-bar-track">
                              <div className="gap-bar-fill" style={{ width: `${pred._barPct}%` }} />
                            </div>
                            <span className="gap-label">+{pred._gapToLeader.toFixed(1)}</span>
                          </>
                        )
                      ) : (
                        <span className="gap-label">&mdash;</span>
                      )}
                    </div>
                    <div
                      className={`col-actual ${errorClass ? `accuracy-${errorClass}` : ''} ${
                        isFiltered ? 'filtered' : ''
                      }`}
                    >
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
                                  Filtered: {pred.filterReason}
                                  <br />
                                  Start: {pred.gridPosition !== null ? pred.gridPosition : 'N/A'}
                                  <br />
                                  Predicted Finish: {displayRank}
                                  <br />
                                  Actual Finish: {pred.actualPosition}
                                  <br />
                                  Error: {posError} position{posError !== 1 ? 's' : ''}
                                </>
                              ) : (
                                <>
                                  Start: {pred.gridPosition !== null ? pred.gridPosition : 'N/A'}
                                  <br />
                                  Predicted Finish: {displayRank}
                                  <br />
                                  Actual Finish: {pred.actualPosition}
                                  <br />
                                  Error: {posError} position{posError !== 1 ? 's' : ''}
                                </>
                              )}
                            </span>
                          </span>
                        </span>
                      ) : (
                        <span className="actual-unknown tooltip-trigger">
                          &mdash;
                          <span className="tooltip">
                            <span className="tooltip-content">No actual result available</span>
                          </span>
                        </span>
                      )}
                    </div>
                  </div>
                );
              });
            })()}
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
    </div>
  );
}

export default PredictionsView;
