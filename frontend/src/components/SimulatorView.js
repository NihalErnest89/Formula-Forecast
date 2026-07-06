import React, { useState, useEffect, useRef, useMemo } from 'react';
import './Simulator.css';
import { DATA_BASE } from '../hooks/useRaces';
import { runMonteCarlo, CHAOS_PRESETS } from '../sim/monteCarlo';
import RaceSelector from './RaceSelector';
import SimControls from './SimControls';
import OddsTable from './OddsTable';

// Pick a sensible default race: the earliest upcoming (future) race, else the
// most recent past race — so the simulator opens on something spoiler-free.
function pickDefaultRace(races) {
  if (!Array.isArray(races) || races.length === 0) return null;
  const future = races
    .filter((r) => r.isFuture)
    .sort((a, b) => a.year - b.year || a.roundNumber - b.roundNumber);
  if (future.length > 0) return future[0];
  return [...races].sort((a, b) => b.year - a.year || b.roundNumber - a.roundNumber)[0];
}

function SimulatorView({ races, uniqueYears }) {
  const [selectedYear, setSelectedYear] = useState(2026);
  const [selectedRace, setSelectedRace] = useState(null);
  const [drivers, setDrivers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [chaos, setChaos] = useState('dry');
  const [dnfEnabled, setDnfEnabled] = useState(false);
  const [runs, setRuns] = useState(10000);
  const [paceOffsets, setPaceOffsets] = useState({});

  const [simResult, setSimResult] = useState(null);
  const [running, setRunning] = useState(false);
  const debounceRef = useRef(null);
  const didDefault = useRef(false);

  // Auto-select a default race once the race list arrives.
  useEffect(() => {
    if (didDefault.current || !races.length) return;
    const def = pickDefaultRace(races);
    if (def) {
      didDefault.current = true;
      setSelectedYear(def.year);
      loadRace(def);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [races]);

  const loadRace = async (race) => {
    try {
      setLoading(true);
      setError(null);
      setSelectedRace(race);
      setPaceOffsets({});
      const response = await fetch(`${DATA_BASE}/predictions/${race.year}-${race.roundNumber}.json`);
      if (!response.ok) throw new Error('Predictions not found');
      const data = await response.json();
      const list = (data.predictionsUnfiltered || data.predictionsFiltered || []).map((p) => ({
        driverName: p.driverName,
        driverNumber: p.driverNumber,
        predictedPosition: p.predictedPosition,
      }));
      setDrivers(list);
    } catch (err) {
      setError('Failed to load race data');
      setDrivers([]);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Re-run the simulation (debounced) whenever inputs change.
  useEffect(() => {
    if (!drivers.length) {
      setSimResult(null);
      return;
    }
    setRunning(true);
    clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      // Defer so the "Simulating…" state can paint before the synchronous run.
      const result = runMonteCarlo(drivers, { runs, chaos, dnfEnabled, paceOffsets });
      setSimResult(result);
      setRunning(false);
    }, 180);
    return () => clearTimeout(debounceRef.current);
  }, [drivers, runs, chaos, dnfEnabled, paceOffsets]);

  const handlePaceChange = (driverName, value) => {
    setPaceOffsets((prev) => ({ ...prev, [driverName]: value }));
  };

  const chaosHint = useMemo(
    () => CHAOS_PRESETS.find((p) => p.key === chaos)?.hint || '',
    [chaos]
  );

  return (
    <div className={`view-body sim-view ${selectedRace ? 'has-sim' : ''}`}>
      {error && <div className="error-message">{error}</div>}

      <RaceSelector
        races={races}
        uniqueYears={uniqueYears}
        selectedRace={selectedRace}
        onSelect={loadRace}
        selectedYear={selectedYear}
        onYearChange={setSelectedYear}
        minimized={!!selectedRace}
        onClear={() => {
          setSelectedRace(null);
          setDrivers([]);
          setSimResult(null);
        }}
        title="Race"
        loading={loading}
      />

      {!selectedRace && !loading && (
        <div className="sim-empty">
          <div className="sim-empty-icon">🏁</div>
          <h2>Race Simulator</h2>
          <p>
            Pick a race to run thousands of Monte Carlo simulations on the model's predictions.
            Tune the weather, reliability and each driver's pace to see how the odds shift.
          </p>
        </div>
      )}

      {loading && (
        <div className="loading-predictions">
          <div className="spinner"></div>
          <p>Loading grid...</p>
        </div>
      )}

      {selectedRace && !loading && drivers.length > 0 && (
        <div className="sim-main">
          <div className="sim-results-panel">
            <div className="sim-results-head">
              <div>
                <h2>{selectedRace.eventName}</h2>
                <p className="sim-subtitle">
                  {selectedRace.year} &middot; Round {selectedRace.roundNumber}
                  {selectedRace.isFuture && <span className="future-badge">Future Race</span>}
                </p>
              </div>
              <div className="sim-meta">
                <span className={`sim-runs-badge ${running ? 'pulsing' : ''}`}>
                  {running ? 'Simulating…' : `${(simResult?.runs || runs).toLocaleString()} sims`}
                </span>
                <span className="sim-chaos-badge">{chaosHint}</span>
              </div>
            </div>

            {simResult && <OddsTable results={simResult.results} running={running} />}

            <p className="sim-disclaimer">
              Odds are Monte Carlo estimates built on the model's predicted positions with added
              race-day variance — not a guarantee. Higher "chaos" widens the spread.
            </p>
          </div>

          <SimControls
            drivers={drivers}
            chaos={chaos}
            onChaosChange={setChaos}
            dnfEnabled={dnfEnabled}
            onDnfToggle={setDnfEnabled}
            runs={runs}
            onRunsChange={setRuns}
            paceOffsets={paceOffsets}
            onPaceChange={handlePaceChange}
            onReset={() => setPaceOffsets({})}
          />
        </div>
      )}
    </div>
  );
}

export default SimulatorView;
