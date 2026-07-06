import { useState, useEffect, useCallback } from 'react';

const DATA_BASE = process.env.PUBLIC_URL + '/data';

/**
 * Loads the shared race list (races.json) once and exposes it plus the
 * descending list of unique years. Used by both the Predictions and Simulator
 * tabs so they stay in sync without re-fetching.
 */
export function useRaces() {
  const [races, setRaces] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

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
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchRaces();
  }, [fetchRaces]);

  const uniqueYears = Array.isArray(races)
    ? [...new Set(races.map((r) => r.year).filter(Boolean))].sort((a, b) => b - a)
    : [];

  return { races, uniqueYears, error, loading };
}

export { DATA_BASE };
