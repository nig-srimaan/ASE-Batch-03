const API_BASE = import.meta.env?.VITE_API_BASE || '';

export async function fetchSystemStatus() {
  const res = await fetch(`${API_BASE}/api/status`);
  return res.json();
}

export async function stepMapekLoop(ignoreCooldown = false) {
  const res = await fetch(`${API_BASE}/api/mapek/step?ignore_cooldown=${ignoreCooldown}`, {
    method: 'POST',
  });
  return res.json();
}

export async function runVivaScenario(scenarioId) {
  const res = await fetch(`${API_BASE}/api/scenarios/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ scenario_id: scenarioId }),
  });
  return res.json();
}

export async function updateWorkload(reqPerMin, noiseStdMs = 1.5) {
  const res = await fetch(`${API_BASE}/api/workload`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ req_per_min: Number(reqPerMin), noise_std_ms: Number(noiseStdMs) }),
  });
  return res.json();
}

export async function updateSlaConfig(payload) {
  const res = await fetch(`${API_BASE}/api/sla`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return res.json();
}

export async function resetSystem(clearHistory = false) {
  const res = await fetch(`${API_BASE}/api/reset?clear_history=${clearHistory}`, {
    method: 'POST',
  });
  return res.json();
}
