import React from 'react';
import { getTeamColorForDriver, getDriverImage } from '../driverData';

// A horizontal probability bar, team-coloured.
function OddsBar({ pct, color }) {
  return (
    <div className="odds-bar-track">
      <div
        className="odds-bar-fill"
        style={{ width: `${Math.max(pct, pct > 0 ? 2 : 0)}%`, background: color }}
      />
      <span className="odds-bar-value">{pct >= 0.1 ? pct.toFixed(1) : '<0.1'}%</span>
    </div>
  );
}

// Compact finishing-position distribution sparkline.
function DistributionSpark({ histogram, color }) {
  const max = Math.max(...histogram, 0.0001);
  return (
    <div className="dist-spark" aria-hidden="true">
      {histogram.map((p, i) => (
        <span
          key={i}
          className="dist-bar"
          style={{ height: `${(p / max) * 100}%`, background: color, opacity: 0.35 + 0.65 * (p / max) }}
          title={`P${i + 1}: ${(p * 100).toFixed(1)}%`}
        />
      ))}
    </div>
  );
}

function OddsTable({ results, running }) {
  return (
    <div className={`odds-table ${running ? 'running' : ''}`}>
      <div className="odds-header">
        <div className="odds-col-driver">Driver</div>
        <div className="odds-col-win">Win</div>
        <div className="odds-col-podium">Podium</div>
        <div className="odds-col-points">Top 5</div>
        <div className="odds-col-exp">Exp.</div>
        <div className="odds-col-dist">Range</div>
      </div>

      {results.map((row, idx) => {
        const team = getTeamColorForDriver(row.driverName);
        const img = getDriverImage(row.driverName);
        return (
          <div
            className="odds-row"
            key={row.driverName}
            style={{
              '--team-color': team.primary,
              '--team-color-secondary': team.secondary,
              animationDelay: `${idx * 0.03}s`,
            }}
          >
            <div className="odds-col-driver">
              <span className="odds-pos">{idx + 1}</span>
              <div className="odds-driver-img">
                {img ? (
                  <img
                    src={img}
                    alt={row.driverName}
                    onError={(e) => {
                      e.target.style.display = 'none';
                    }}
                  />
                ) : (
                  <span className="odds-driver-num">{row.driverNumber || '?'}</span>
                )}
              </div>
              <span className="odds-driver-name">{row.driverName}</span>
            </div>
            <div className="odds-col-win">
              <OddsBar pct={row.winPct} color={team.primary} />
            </div>
            <div className="odds-col-podium">
              <OddsBar pct={row.podiumPct} color={team.primary} />
            </div>
            <div className="odds-col-points">
              <OddsBar pct={row.top5Pct} color={team.primary} />
            </div>
            <div className="odds-col-exp">P{row.expectedFinish.toFixed(1)}</div>
            <div className="odds-col-dist">
              <DistributionSpark histogram={row.histogram} color={team.primary} />
              <span className="dist-range">
                P{row.best}&ndash;P{row.worst}
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default OddsTable;
