import React, { useState, useEffect, useCallback } from 'react';
import { getTeamColorForDriver, getDriverImage, driverInfo } from '../driverData';
import { DATA_BASE } from '../hooks/useRaces';

/**
 * Projected championship standings: actual points from completed races plus
 * model-predicted points (25-18-15-...-1) for every remaining race.
 * Reads the precomputed standings/<year>.json files.
 */
function StandingsView({ uniqueYears }) {
  const defaultYear = uniqueYears.length ? Math.max(...uniqueYears) : 2026;
  const [selectedYear, setSelectedYear] = useState(defaultYear);
  const [standings, setStandings] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const fetchStandings = useCallback(async (year) => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch(`${DATA_BASE}/standings/${year}.json`);
      if (!response.ok) throw new Error('Standings not found');
      setStandings(await response.json());
    } catch (err) {
      console.error('Error fetching standings:', err);
      setError('Failed to load standings');
      setStandings(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStandings(selectedYear);
  }, [selectedYear, fetchStandings]);

  // In-season: movement from current standings to the projection.
  // Completed season: how the driver finished vs where the model ranked them
  // (green = outperformed the model's prediction).
  const movement = (delta) => {
    if (delta > 0) return <span className="standings-move up">▲ {delta}</span>;
    if (delta < 0) return <span className="standings-move down">▼ {-delta}</span>;
    return <span className="standings-move flat">–</span>;
  };

  const fmtPts = (v) => (Number.isInteger(v) ? v : v.toFixed(1));

  return (
    <div className="view-body standings-view">
      {error && <div className="error-message">{error}</div>}

      <div className="standings-header">
        <h2>Championship Projection</h2>
        <div className="year-buttons">
          {uniqueYears.map((year) => (
            <button
              key={year}
              className={`year-button ${selectedYear === year ? 'active' : ''}`}
              onClick={() => setSelectedYear(year)}
            >
              {year}
            </button>
          ))}
        </div>
      </div>

      {loading && (
        <div className="loading-predictions">
          <div className="spinner"></div>
          <p>Loading standings...</p>
        </div>
      )}

      {standings && !loading && (
        <div className="standings-container">
          <p className="standings-subtitle">
            {standings.seasonComplete
              ? `Final standings after ${standings.completedRounds} rounds — compared against the championship the model predicted race by race`
              : `${standings.completedRounds} rounds complete + ${standings.remainingRounds} predicted by the model`}
          </p>

          <div className="standings-table">
            <div className="standings-row standings-table-header">
              <span>Pos</span>
              <span>Driver</span>
              <span className="num">Points</span>
              <span className="num">{standings.seasonComplete ? 'Model Pts' : '+ Predicted'}</span>
              <span className="num">{standings.seasonComplete ? 'Model Pos' : 'Projected'}</span>
              <span className="num">{standings.seasonComplete ? 'vs Model' : 'Move'}</span>
            </div>

            {standings.standings.map((row) => (
              <div
                key={row.driverNumber}
                className="standings-row"
                style={{ borderLeft: `4px solid ${getTeamColorForDriver(row.driverName) || '#666'}` }}
              >
                <span className="standings-pos">{row.projectedRank}</span>
                <span className="standings-driver">
                  {getDriverImage(row.driverName) && (
                    <img src={getDriverImage(row.driverName)} alt="" className="standings-driver-img" />
                  )}
                  <span>
                    <strong>{driverInfo[row.driverName]?.fullName || row.driverName}</strong>
                    <span className="standings-team">{row.teamName}</span>
                  </span>
                </span>
                <span className="num">{fmtPts(row.currentPoints)}</span>
                {standings.seasonComplete ? (
                  <>
                    <span className="num standings-predicted">{fmtPts(row.modelPoints)}</span>
                    <span className="num standings-projected">P{row.modelRank}</span>
                    {movement(row.modelRank - row.projectedRank)}
                  </>
                ) : (
                  <>
                    <span className="num standings-predicted">+{fmtPts(row.predictedPoints)}</span>
                    <span className="num standings-projected">{fmtPts(row.projectedTotal)}</span>
                    {movement(row.currentRank - row.projectedRank)}
                  </>
                )}
              </div>
            ))}
          </div>

          <p className="standings-note">{standings.note}</p>
        </div>
      )}
    </div>
  );
}

export default StandingsView;
