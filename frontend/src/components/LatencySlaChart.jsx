import React from 'react';

export default function LatencySlaChart({ history = [], slaLimit = 100, hysteresisMargin = 12 }) {
  const width = 760;
  const height = 230;
  const pad = { top: 20, right: 24, bottom: 32, left: 46 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const maxY = Math.max(150, slaLimit + 35, ...history.map((d) => d.latency_ms || 0));

  const yPos = (val) => pad.top + plotH - Math.min(1, Math.max(0, val / maxY)) * plotH;
  const xPos = (idx) =>
    history.length <= 1
      ? pad.left + plotW / 2
      : pad.left + (idx / (history.length - 1)) * plotW;

  const points = history.map((d, i) => `${xPos(i)},${yPos(d.latency_ms)}`).join(' ');
  const slaY = yPos(slaLimit);
  const hystY = yPos(Math.max(0, slaLimit - hysteresisMargin));

  return (
    <div className="chart-card">
      <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', height: 'auto' }}>
        <line x1={pad.left} y1={slaY} x2={width - pad.right} y2={slaY} stroke="#ef4444" strokeDasharray="6 4" strokeWidth="2" />
        <text x={width - pad.right - 4} y={slaY - 6} fill="#f87171" fontSize="11" textAnchor="end">
          Latency SLA Ceiling ({slaLimit} ms)
        </text>
        <line x1={pad.left} y1={hystY} x2={width - pad.right} y2={hystY} stroke="#10b981" strokeDasharray="4 4" strokeWidth="1.5" />
        <text x={width - pad.right - 4} y={hystY + 13} fill="#34d399" fontSize="10" textAnchor="end">
          Hysteresis Recovery Ceiling ({slaLimit - hysteresisMargin} ms)
        </text>
        {history.length > 1 && (
          <polyline fill="none" stroke="#38bdf8" strokeWidth="2.5" points={points} />
        )}
        {history.map((d, i) => (
          <circle
            key={i}
            cx={xPos(i)}
            cy={yPos(d.latency_ms)}
            r={d.event_tag === 'SWITCHED_ACTIVE_MODEL' ? 6 : 3.5}
            fill={d.latency_ms > slaLimit ? '#ef4444' : '#38bdf8'}
          />
        ))}
      </svg>
    </div>
  );
}
