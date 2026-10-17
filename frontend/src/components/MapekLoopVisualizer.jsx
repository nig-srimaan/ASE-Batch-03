import React from 'react';

export default function MapekLoopVisualizer({ trace }) {
  if (!trace || !trace.monitor) return null;
  const phases = [
    { key: 'M', title: 'Monitor', detail: `${trace.monitor.active_model_name} @ ${trace.monitor.latest_latency_ms}ms | CPU ${trace.monitor.cpu_percent}%` },
    { key: 'A', title: 'Analyze', detail: `${trace.analyze.stability_state} (${trace.analyze.consecutive_violations} breaches / ${trace.analyze.consecutive_stable_samples} stable)` },
    { key: 'P', title: 'Plan', detail: trace.plan.selection_rationale },
    { key: 'E', title: 'Execute', detail: `${trace.execute.action}: ${trace.execute.reason || ''}` },
  ];
  return (
    <div className="mapek-grid">
      {phases.map((p) => (
        <div key={p.key} className="mapek-step-card">
          <div className="mapek-badge">{p.key}</div>
          <h4>{p.title}</h4>
          <p>{p.detail}</p>
        </div>
      ))}
    </div>
  );
}
