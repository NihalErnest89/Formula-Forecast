import React from 'react';

const TABS = [
  { key: 'predictions', label: 'Predictions' },
  { key: 'standings', label: 'Standings' },
  { key: 'simulator', label: 'Simulator' },
  { key: 'about', label: 'About' },
];

function NavTabs({ active, onChange }) {
  return (
    <nav className="nav-tabs" role="tablist" aria-label="Main">
      <div className="nav-tabs-inner">
        {TABS.map((tab) => (
          <button
            key={tab.key}
            role="tab"
            aria-selected={active === tab.key}
            className={`nav-tab ${active === tab.key ? 'active' : ''}`}
            onClick={() => onChange(tab.key)}
          >
            {tab.label}
          </button>
        ))}
      </div>
    </nav>
  );
}

export default NavTabs;
