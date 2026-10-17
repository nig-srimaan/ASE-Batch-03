"""
Component 5: Runtime Monitoring Module (with EWMA Trend Forecaster)
===================================================================
Responsibilities:
- Continuously collects runtime operational metrics:
  * Prediction latency (latest, moving average, p50, p95, p99)
  * EWMA short-horizon latency forecast & linear trend slope (ms/tick)
  * CPU utilization (%) and Memory utilization (%)
  * Workload / Request rate (requests/min)
  * Active model prediction confidence & drift indicator
- Exports Prometheus-compatible metrics at `/metrics`
"""
import time
import numpy as np
from collections import deque
from threading import Lock
from typing import Dict, Any, Optional, Tuple

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class RuntimeMonitoringModule:
    """Collects sliding-window runtime telemetry, EWMA forecasts, and Prometheus metrics."""

    def __init__(self, window_size: int = 10, ewma_alpha: float = 0.45):
        self.window_size = window_size
        self.ewma_alpha = ewma_alpha
        self.latencies_ms = deque(maxlen=window_size)
        self.confidences = deque(maxlen=window_size)
        self.cpu_samples = deque(maxlen=window_size)
        self.workload_req_per_min: float = 60.0
        self.noise_std_ms: float = 1.5
        self.total_predictions: int = 0
        self.last_latency_ms: float = 80.0
        self.ewma_latency_ms: float = 80.0
        self.last_cpu_percent: float = 65.0
        self.last_memory_percent: float = 42.0
        self._lock = Lock()

    def _read_host_memory_percent(self) -> float:
        if HAS_PSUTIL:
            try:
                return float(psutil.virtual_memory().percent)
            except Exception:
                pass
        try:
            with open("/proc/meminfo", "r") as f:
                info = {}
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        info[parts[0].strip()] = int(parts[1].strip().split()[0])
                total = info.get("MemTotal", 1)
                avail = info.get("MemAvailable", total // 2)
                return round((1.0 - avail / total) * 100.0, 1)
        except Exception:
            return 42.5

    def compute_realistic_telemetry(
        self,
        model_profile: Dict[str, Any],
        explicit_latency_ms: Optional[float] = None,
        workload_override: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        workload = workload_override if workload_override is not None else self.workload_req_per_min
        base_lat = float(model_profile["baseline_latency_ms"])
        base_cpu = float(model_profile["baseline_cpu_percent"])
        lat_sens = float(model_profile.get("latency_load_sensitivity", 0.3))
        cpu_sens = float(model_profile.get("cpu_load_sensitivity", 0.12))

        excess_load = max(0.0, workload - 60.0)
        under_load = min(0.0, workload - 60.0)

        if explicit_latency_ms is not None:
            latency_ms = float(explicit_latency_ms)
            cpu_pct = base_cpu + (latency_ms - base_lat) * 0.25
        else:
            jitter = float(np.random.normal(0.0, self.noise_std_ms)) if self.noise_std_ms > 0 else 0.0
            latency_ms = max(5.0, base_lat + excess_load * lat_sens + under_load * (lat_sens * 0.25) + jitter)
            cpu_pct = base_cpu + excess_load * cpu_sens + under_load * (cpu_sens * 0.2)

        cpu_pct = float(np.clip(cpu_pct, 10.0, 98.0))
        mem_pct = self._read_host_memory_percent()
        return round(latency_ms, 2), round(cpu_pct, 1), round(mem_pct, 1)

    def record_observation(
        self,
        latency_ms: float,
        cpu_percent: float,
        memory_percent: float,
        confidence: float = 0.91,
    ):
        with self._lock:
            val = float(latency_ms)
            if not self.latencies_ms:
                self.ewma_latency_ms = val
            else:
                self.ewma_latency_ms = self.ewma_alpha * val + (1.0 - self.ewma_alpha) * self.ewma_latency_ms
            self.latencies_ms.append(val)
            self.cpu_samples.append(float(cpu_percent))
            self.confidences.append(float(confidence))
            self.last_latency_ms = val
            self.last_cpu_percent = float(cpu_percent)
            self.last_memory_percent = float(memory_percent)
            self.total_predictions += 1

    def set_workload(self, req_per_min: float, noise_std_ms: float = 1.5):
        with self._lock:
            self.workload_req_per_min = float(req_per_min)
            self.noise_std_ms = float(noise_std_ms)

    def reset_window(self, initial_latency_ms: float = 80.0, initial_cpu_percent: float = 65.0):
        with self._lock:
            self.latencies_ms.clear()
            self.cpu_samples.clear()
            self.confidences.clear()
            self.latencies_ms.append(initial_latency_ms)
            self.cpu_samples.append(initial_cpu_percent)
            self.last_latency_ms = initial_latency_ms
            self.ewma_latency_ms = initial_latency_ms
            self.last_cpu_percent = initial_cpu_percent

    def get_metrics_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            lats = list(self.latencies_ms)
            cpus = list(self.cpu_samples)
            confs = list(self.confidences)
            workload = self.workload_req_per_min
            total_reqs = self.total_predictions
            last_lat = self.last_latency_ms
            ewma_lat = self.ewma_latency_ms
            last_cpu = self.last_cpu_percent
            last_mem = self.last_memory_percent

        if not lats:
            lats = [last_lat]
        if not cpus:
            cpus = [last_cpu]

        avg_lat = float(np.mean(lats))
        p50_lat = float(np.percentile(lats, 50))
        p95_lat = float(np.percentile(lats, 95))
        p99_lat = float(np.percentile(lats, 99))
        avg_cpu = float(np.mean(cpus))
        avg_conf = float(np.mean(confs)) if confs else 0.90

        # Compute short-term linear trend slope (ms per tick)
        if len(lats) >= 3:
            x = np.arange(len(lats), dtype=np.float64)
            y = np.array(lats, dtype=np.float64)
            slope = float(np.polyfit(x, y, 1)[0])
        else:
            slope = 0.0

        forecast_next_ms = max(5.0, ewma_lat + 0.6 * slope)

        return {
            "latest_latency_ms": round(last_lat, 2),
            "ewma_latency_ms": round(ewma_lat, 2),
            "forecast_next_latency_ms": round(forecast_next_ms, 2),
            "latency_trend_slope_ms": round(slope, 2),
            "avg_latency_ms": round(avg_lat, 2),
            "p50_latency_ms": round(p50_lat, 2),
            "p95_latency_ms": round(p95_lat, 2),
            "p99_latency_ms": round(p99_lat, 2),
            "recent_latencies_ms": [round(x, 1) for x in lats[-8:]],
            "cpu_percent": round(last_cpu, 1),
            "avg_cpu_percent": round(avg_cpu, 1),
            "memory_percent": round(last_mem, 1),
            "workload_req_per_min": round(workload, 1),
            "avg_prediction_confidence": round(avg_conf, 4),
            "window_sample_count": len(lats),
            "total_predictions_served": total_reqs,
        }

    def export_prometheus_metrics(self, active_model_id: str, sla_violated: bool, switch_count: int) -> str:
        m = self.get_metrics_snapshot()
        lines = [
            "# HELP adaptive_ml_latest_latency_ms Latest inference latency in milliseconds",
            "# TYPE adaptive_ml_latest_latency_ms gauge",
            f'adaptive_ml_latest_latency_ms{{model="{active_model_id}"}} {m["latest_latency_ms"]}',
            "# HELP adaptive_ml_ewma_forecast_latency_ms EWMA short-horizon forecasted latency in milliseconds",
            "# TYPE adaptive_ml_ewma_forecast_latency_ms gauge",
            f'adaptive_ml_ewma_forecast_latency_ms{{model="{active_model_id}"}} {m["forecast_next_latency_ms"]}',
            "# HELP adaptive_ml_p95_latency_ms 95th percentile sliding-window latency in milliseconds",
            "# TYPE adaptive_ml_p95_latency_ms gauge",
            f'adaptive_ml_p95_latency_ms{{model="{active_model_id}"}} {m["p95_latency_ms"]}',
            "# HELP adaptive_ml_cpu_utilization_percent Current CPU utilization percentage",
            "# TYPE adaptive_ml_cpu_utilization_percent gauge",
            f'adaptive_ml_cpu_utilization_percent{{model="{active_model_id}"}} {m["cpu_percent"]}',
            "# HELP adaptive_ml_workload_req_per_min Incoming request rate per minute",
            "# TYPE adaptive_ml_workload_req_per_min gauge",
            f'adaptive_ml_workload_req_per_min {m["workload_req_per_min"]}',
            "# HELP adaptive_ml_sla_violated Whether the active model currently breaches the SLA (1=yes, 0=no)",
            "# TYPE adaptive_ml_sla_violated gauge",
            f'adaptive_ml_sla_violated {1 if sla_violated else 0}',
            "# HELP adaptive_ml_model_switches_total Total number of MAPE-K runtime model adaptations",
            "# TYPE adaptive_ml_model_switches_total counter",
            f'adaptive_ml_model_switches_total {switch_count}',
        ]
        return "\n".join(lines) + "\n"


runtime_monitor = RuntimeMonitoringModule()
