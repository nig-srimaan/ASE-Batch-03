import React, { useEffect, useState } from 'react';
import {
  fetchSystemStatus,
  stepMapekLoop,
  runVivaScenario,
  resetSystem,
} from './api.js';
import LatencySlaChart from './components/LatencySlaChart.jsx';
import MapekLoopVisualizer from './components/MapekLoopVisualizer.jsx';
import VivaScenarioRunner from './components/VivaScenarioRunner.jsx';
import ComponentInspector from './components/ComponentInspector.jsx';

export default function App() {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);

  const refresh = async () => {
    const data = await fetchSystemStatus();
    setStatus(data);
  };

  useEffect(() => {
    refresh();
  }, []);

  const handleScenario = async (id) => {
    setLoading(true);
    try {
      const res = await runVivaScenario(id);
      setStatus(res.system_status);
    } finally {
      setLoading(false);
    }
  };

  if (!status) return <div style={{ padding: 32, color: '#e2e8f0' }}>Loading Adaptive ML MAPE-K Dashboard...</div>;

  return (
    <div style={{ maxWidth: 1280, margin: '0 auto', padding: 24, fontFamily: 'Inter, system-ui, sans-serif' }}>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
        <div>
          <h1 style={{ margin: 0 }}>Self-Adaptive ML Model-Switching System (MAPE-K)</h1>
          <p style={{ margin: '4px 0 0', opacity: 0.8 }}>
            12-Component Architecture with Persistent SLA Detection, Multi-Criteria Selection & Bidirectional Hysteresis
          </p>
        </div>
        <div style={{ display: 'flex', gap: 10 }}>
          <button onClick={() => stepMapekLoop(false).then(refresh)}>Step MAPE-K Cycle</button>
          <button onClick={() => resetSystem(true).then((s) => setStatus(s))}>Reset System</button>
        </div>
      </header>

      <LatencySlaChart
        history={status.telemetry_history}
        slaLimit={status.sla.thresholds.max_latency_sla_ms}
        hysteresisMargin={status.stability.hysteresis_latency_margin_ms}
      />

      <MapekLoopVisualizer trace={status.last_mapek_trace} />
      <VivaScenarioRunner onRunScenario={handleScenario} loading={loading} />
      <ComponentInspector components={status.components_registry} />
    </div>
  );
}
