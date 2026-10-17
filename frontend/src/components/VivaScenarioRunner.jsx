import React from 'react';

const SCENARIOS = [
  {
    id: 'temporary_fluctuation',
    title: 'Case 1: Temporary Fluctuation (No Switch)',
    subtitle: '90 → 105 → 98 → 103 → 95 ms (Absorbed by Stability Manager)',
  },
  {
    id: 'persistent_violation',
    title: 'Case 2: Persistent SLA Violation (Switch)',
    subtitle: '105 → 112 → 120 → 130 → 125 ms (Switches XGBoost → Random Forest)',
  },
  {
    id: 'accuracy_sla_guard',
    title: 'Case 3: Accuracy SLA Floor Guard (85%)',
    subtitle: 'Rejects Logistic Regression (82% Acc < 85% SLA) despite 20ms latency',
  },
  {
    id: 'cooldown_recovery',
    title: 'Case 4: Hysteresis & Cooldown Recovery',
    subtitle: 'Wait for stability window + hysteresis margin → Switches RF → XGBoost',
  },
  {
    id: 'full_lifecycle',
    title: 'Run Full Bidirectional Lifecycle (15 Ticks)',
    subtitle: 'Executes Case 1 → Case 2 → Case 4 sequentially',
  },
];

export default function VivaScenarioRunner({ onRunScenario, loading }) {
  return (
    <div className="scenario-list">
      {SCENARIOS.map((s) => (
        <button
          key={s.id}
          disabled={loading}
          onClick={() => onRunScenario(s.id)}
          className="scenario-btn"
        >
          <strong>{s.title}</strong>
          <span>{s.subtitle}</span>
        </button>
      ))}
    </div>
  );
}
