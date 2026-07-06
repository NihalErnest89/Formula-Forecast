import React from 'react';
import { CHAOS_PRESETS } from '../sim/monteCarlo';
import { getTeamColorForDriver } from '../driverData';

const RUN_OPTIONS = [1000, 10000, 50000];

function SimControls({
  drivers,
  chaos,
  onChaosChange,
  dnfEnabled,
  onDnfToggle,
  runs,
  onRunsChange,
  paceOffsets,
  onPaceChange,
  onReset,
}) {
  return (
    <div className="sim-controls">
      <div className="sim-control-group">
        <span className="sim-control-label">Conditions</span>
        <div className="chaos-presets">
          {CHAOS_PRESETS.map((preset) => (
            <button
              key={preset.key}
              className={`chaos-btn ${chaos === preset.key ? 'active' : ''}`}
              onClick={() => onChaosChange(preset.key)}
              title={preset.hint}
            >
              {preset.label}
            </button>
          ))}
        </div>
      </div>

      <div className="sim-control-group">
        <span className="sim-control-label">Reliability</span>
        <label className="dnf-toggle">
          <input type="checkbox" checked={dnfEnabled} onChange={(e) => onDnfToggle(e.target.checked)} />
          <span>Allow DNFs</span>
        </label>
      </div>

      <div className="sim-control-group">
        <span className="sim-control-label">Simulations</span>
        <div className="runs-presets">
          {RUN_OPTIONS.map((opt) => (
            <button
              key={opt}
              className={`runs-btn ${runs === opt ? 'active' : ''}`}
              onClick={() => onRunsChange(opt)}
            >
              {opt >= 1000 ? `${opt / 1000}k` : opt}
            </button>
          ))}
        </div>
      </div>

      <div className="sim-control-group sim-pace-group">
        <div className="sim-pace-head">
          <span className="sim-control-label">Pace tuning</span>
          <button className="pace-reset-btn" onClick={onReset}>
            Reset
          </button>
        </div>
        <p className="sim-pace-hint">Drag left for a faster day, right for a slower one.</p>
        <div className="pace-sliders">
          {drivers.map((d) => {
            const team = getTeamColorForDriver(d.driverName);
            const val = paceOffsets[d.driverName] || 0;
            return (
              <div className="pace-slider-row" key={d.driverName}>
                <span className="pace-driver" style={{ color: team.primary }}>
                  {d.driverName}
                </span>
                <input
                  type="range"
                  min="-3"
                  max="3"
                  step="0.5"
                  value={val}
                  onChange={(e) => onPaceChange(d.driverName, parseFloat(e.target.value))}
                  style={{ accentColor: team.primary }}
                />
                <span className={`pace-val ${val !== 0 ? 'changed' : ''}`}>
                  {val > 0 ? `+${val}` : val}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

export default SimControls;
