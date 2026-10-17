import os
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class SystemConfig:
    """Centralized configuration for the 12-Component Self-Adaptive ML System."""
    project_name: str = "Self-Adaptive ML Model-Switching System (MAPE-K)"
    version: str = "2.0.0"

    # SLA Default Thresholds (Component 6: SLA Manager)
    min_accuracy_sla: float = float(os.getenv("SLA_MIN_ACCURACY", "0.85"))      # >= 85%
    max_latency_sla_ms: float = float(os.getenv("SLA_MAX_LATENCY_MS", "100.0")) # <= 100 ms
    max_cpu_sla_percent: float = float(os.getenv("SLA_MAX_CPU_PERCENT", "70.0"))# <= 70%
    allow_emergency_degrade_below_accuracy_sla: bool = False

    # Stability & Hysteresis Defaults (Component 9: Stability & Hysteresis Manager)
    persistent_violation_window: int = int(os.getenv("PERSISTENT_VIOLATION_WINDOW", "5"))
    recovery_stability_window: int = int(os.getenv("RECOVERY_STABILITY_WINDOW", "5"))
    cooldown_seconds: float = float(os.getenv("COOLDOWN_SECONDS", "10.0"))
    hysteresis_latency_margin_ms: float = float(os.getenv("HYSTERESIS_MARGIN_MS", "12.0"))
    min_accuracy_gain_for_upgrade: float = 0.02

    # Runtime Monitoring Defaults (Component 5)
    telemetry_window_size: int = 20
    mapek_loop_interval_sec: float = 2.0

    # Paths & Persistence (Component 11: Knowledge Base)
    base_dir: str = field(default_factory=lambda: os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    database_url: str = field(default_factory=lambda: os.getenv(
        "DATABASE_URL",
        f"sqlite:///{os.path.abspath(os.path.join(os.path.dirname(__file__), '../../saved_models/knowledge_base.db'))}"
    ))

    @property
    def models_dir(self) -> str:
        path = os.path.join(self.base_dir, "saved_models")
        os.makedirs(path, exist_ok=True)
        return path

config = SystemConfig()
