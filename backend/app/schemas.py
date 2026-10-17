from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class PredictionRequest(BaseModel):
    features: Optional[List[float]] = Field(
        default=None,
        description="20 numerical cybersecurity network flow features. If omitted, a realistic sample flow is generated."
    )
    preset_id: Optional[str] = Field(
        default=None,
        description="Optional preset traffic profile: 'normal_web_https', 'syn_flood_ddos', 'port_scan_probe', 'benign_file_transfer'"
    )
    workload_multiplier: Optional[float] = Field(
        default=None,
        description="Optional override for current workload intensity (1.0 = normal, 1.8 = high load)."
    )
    induced_latency_ms: Optional[float] = Field(
        default=None,
        description="Optional explicit latency injection (ms) for deterministic SLA testing."
    )

class PredictionResponse(BaseModel):
    prediction: int
    label: str
    confidence: float
    probabilities: List[float]
    served_by_model: str
    model_display_name: str
    model_accuracy: float
    latency_ms: float
    cpu_percent: float
    workload_req_per_min: float
    sla_satisfied: bool
    active_sla_target_ms: float
    preset_used: Optional[str] = None
    top_feature_attributions: List[Dict[str, Any]] = []

class SLAConfigUpdate(BaseModel):
    min_accuracy_sla: Optional[float] = Field(None, ge=0.50, le=0.99)
    max_latency_sla_ms: Optional[float] = Field(None, ge=15.0, le=500.0)
    max_cpu_sla_percent: Optional[float] = Field(None, ge=20.0, le=99.0)
    allow_emergency_degrade_below_accuracy_sla: Optional[bool] = None

class StabilityConfigUpdate(BaseModel):
    persistent_violation_window: Optional[int] = Field(None, ge=1, le=20)
    recovery_stability_window: Optional[int] = Field(None, ge=1, le=20)
    cooldown_seconds: Optional[float] = Field(None, ge=0.0, le=120.0)
    hysteresis_latency_margin_ms: Optional[float] = Field(None, ge=0.0, le=50.0)

class WorkloadUpdate(BaseModel):
    req_per_min: float = Field(..., ge=10.0, le=500.0)
    noise_std_ms: float = Field(default=3.0, ge=0.0, le=25.0)

class ScenarioRunRequest(BaseModel):
    scenario_id: str = Field(
        ...,
        description="One of: 'temporary_fluctuation', 'persistent_violation', 'accuracy_sla_guard', 'cooldown_recovery', 'full_lifecycle'"
    )
