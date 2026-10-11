import React, { useState } from 'react';
import './App.css';
import { useRaces } from './hooks/useRaces';
import NavTabs from './components/NavTabs';
import PredictionsView from './components/PredictionsView';
import SimulatorView from './components/SimulatorView';
import StandingsView from './components/StandingsView';

function AboutView() {
  return (
    <div className="view-body about-view">
      <div className="about-card">
        <h2>About Formula Forecast</h2>
        <p>
          Formula Forecast predicts F1 finishing order with two neural networks trained on FastF1
          data from 2018&ndash;2025 and tested on the 2026 season, which they never see in training.
          Both use six inputs: the starting grid, share of the championship leader's points, recent
          form (crashes excluded), constructor standing, track history and track type.
        </p>
        <p>
          <strong>Two models, one per stage of the weekend.</strong> Once qualifying is in, the
          post-qualifying model uses the real starting grid. Before that, the pre-qualifying model
          uses the driver's average grid slot this season instead, so its predictions are
          projections. The site shows every completed race and the next one, which switches to the
          post-qualifying model when its qualifying is saved.
        </p>
        <p>
          <strong>How it compares to the obvious baseline:</strong> on 2026 races, its predicted top
          10 is 0.09 places closer to the result on average than simply using qualifying order, and
          gets the exact position more often (39% vs 35%); within one place the two are level (75%).
          Excluding DNFs and large grid drops. Each prediction is the average of five networks
          trained with different random seeds, so it doesn't hinge on one lucky or unlucky run.
        </p>
        <h3>Predictions</h3>
        <p>
          Browse any race and compare the model's predicted order against the actual result, with
          per-driver accuracy and an optional filtered view that excludes DNFs and large grid drops.
        </p>
        <h3>Standings</h3>
        <p>
          Projected championship standings: points already scored in completed races, plus
          points from the pre-qualifying model's projection of every remaining race (as if the
          season continues as it is), accumulated into a final championship order. Movement arrows show how the projection differs from the current standings.
        </p>
        <h3>Simulator</h3>
        <p>
          Each prediction stores a continuous pace score per driver. The simulator treats that score
          as the centre of a driver's race-day pace and runs thousands of Monte Carlo races &mdash;
          adding variance for weather and reliability &mdash; to estimate Win, Podium and Points
          probabilities. Drag the pace sliders for instant "what-if" scenarios.
        </p>
      </div>
    </div>
  );
}

function App() {
  const [activeTab, setActiveTab] = useState('predictions');
  const { races, uniqueYears, error } = useRaces();

  return (
    <div className="App">
      <header className="App-header">
        <div className="header-stripe" aria-hidden="true"></div>
        <h1>Formula Forecast</h1>
        <p>F1 Race Position Predictions</p>
      </header>

      <NavTabs active={activeTab} onChange={setActiveTab} />

      <main className="App-main">
        {error && <div className="error-message">{error}</div>}

        {activeTab === 'predictions' && (
          <PredictionsView races={races} uniqueYears={uniqueYears} />
        )}
        {activeTab === 'standings' && (
          <StandingsView uniqueYears={uniqueYears} />
        )}
        {activeTab === 'simulator' && (
          <SimulatorView races={races} uniqueYears={uniqueYears} />
        )}
        {activeTab === 'about' && <AboutView />}
      </main>

      <footer className="App-footer">
        <p>Neural networks trained on FastF1 data (2018&ndash;2025) &middot; tested on 2026</p>
        <p>Created by Nihal Ernest</p>
      </footer>
    </div>
  );
}

export default App;
