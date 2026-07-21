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
          Formula Forecast predicts F1 finishing positions with a deep neural network trained on
          FastF1 timing data (2020&ndash;2024). The model is a regularized 3-layer MLP using Dropout,
          Batch Normalization and weight decay, optimized with a position-aware Huber loss.
        </p>
        <h3>Predictions</h3>
        <p>
          Browse any race and compare the model's predicted order against the actual result, with
          per-driver accuracy and an optional filtered view that excludes DNFs and large grid drops.
        </p>
        <h3>Standings</h3>
        <p>
          Projected championship standings: points already scored in completed races, plus
          model-predicted points for every remaining race, accumulated into a final championship
          order. Movement arrows show how the projection differs from the current standings.
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
        <p>Powered by deep neural networks trained on FastF1 data (2020-2024)</p>
        <p>Created by Nihal Ernest</p>
      </footer>
    </div>
  );
}

export default App;
