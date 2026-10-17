"""
Component 6: SLA Manager
========================
Responsibilities:
- Defines acceptable operational performance boundaries:
  * Accuracy >= 85% (0.85)
  * Prediction Latency <= 100 ms
  * CPU Utilization <= 70%
- Evaluates whether the active model currently satisfies all SLA rules
- Projects candidate model latency & CPU under current workload conditions so the
  Adaptive Model Selector can verify whether an alternative model will satisfy the SLA
"""
from typing import Dict, Any, List
from backend.app.config import config


class SLAManagerModule:
    """Manages Service Level Agreement thresholds and compliance verification."""

    def __init__(self):
        self.min_accuracy_sla: float = config.min_accuracy_sla
        self.max_latency_sla_ms: float = config.max_latency_sla_ms
        self.max_cpu_sla_percent: float = config.max_cpu_sla_percent
        self.allow_emergency_degrade_below_accuracy_sla: bool = config.allow_emergency_degrade_below_accuracy_sla

    def update_sla(
        self,
        min_accuracy_sla: float = None,
        max_latency_sla_ms: float = None,
        max_cpu_sla_percent: float = None,
        allow_emergency_degrade: bool = None,
    ) -> Dict[str, Any]:
        if min_accuracy_sla is not None:
            self.min_accuracy_sla = round(float(min_accuracy_sla), 4)
        if max_latency_sla_ms is not None:
            self.max_latency_sla_ms = round(float(max_latency_sla_ms), 2)
        if max_cpu_sla_percent is not None:
            self.max_cpu_sla_percent = round(float(max_cpu_sla_percent), 2)
        if allow_emergency_degrade is not None:
            self.allow_emergency_degrade_below_accuracy_sla = bool(allow_emergency_degrade)
        return self.get_sla_spec()

    def get_sla_spec(self) -> Dict[str, Any]:
        return {
            "min_accuracy_sla": self.min_accuracy_sla,
            "min_accuracy_percent": round(self.min_accuracy_sla * 100.0, 1),
            "max_latency_sla_ms": self.max_latency_sla_ms,
            "max_cpu_sla_percent": self.max_cpu_sla_percent,
            "allow_emergency_degrade_below_accuracy_sla": self.allow_emergency_degrade_below_accuracy_sla,
        }

    def evaluate_current_compliance(
        self,
        active_profile: Dict[str, Any],
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Checks if the active model & current runtime metrics satisfy all SLA rules."""
        observed_latency = float(metrics["latest_latency_ms"])
        observed_cpu = float(metrics["cpu_percent"])
        model_accuracy = float(active_profile["accuracy"])

        latency_ok = observed_latency <= self.max_latency_sla_ms
        cpu_ok = observed_cpu <= self.max_cpu_sla_percent
        accuracy_ok = model_accuracy >= self.min_accuracy_sla

        violations: List[str] = []
        if not latency_ok:
            violations.append(f"Latency {observed_latency:.1f}ms > {self.max_latency_sla_ms:.1f}ms SLA")
        if not cpu_ok:
            violations.append(f"CPU {observed_cpu:.1f}% > {self.max_cpu_sla_percent:.1f}% SLA")
        if not accuracy_ok:
            violations.append(f"Accuracy {model_accuracy*100:.1f}% < {self.min_accuracy_sla*100:.1f}% SLA")

        is_satisfied = latency_ok and cpu_ok and accuracy_ok
        return {
            "sla_satisfied": is_satisfied,
            "latency_ok": latency_ok,
            "cpu_ok": cpu_ok,
            "accuracy_ok": accuracy_ok,
            "observed_latency_ms": observed_latency,
            "observed_cpu_percent": observed_cpu,
            "model_accuracy": model_accuracy,
            "violations": violations,
            "thresholds": self.get_sla_spec(),
        }

    def project_candidate_performance(
        self,
        candidate_profile: Dict[str, Any],
        current_workload_req_per_min: float,
        active_profile: Dict[str, Any],
        observed_latency_ms: float,
    ) -> Dict[str, Any]:
        """
        Estimates the latency and CPU that `candidate_profile` will experience under the
        current operational load so the planner can verify SLA feasibility before switching.
        """
        base_lat = float(candidate_profile["baseline_latency_ms"])
        base_cpu = float(candidate_profile["baseline_cpu_percent"])
        lat_sens = float(candidate_profile.get("latency_load_sensitivity", 0.25))
        cpu_sens = float(candidate_profile.get("cpu_load_sensitivity", 0.12))

        # Infer effective load pressure from either explicit workload_req_per_min or observed latency ratio
        active_base_lat = max(1.0, float(active_profile["baseline_latency_ms"]))
        stress_ratio = max(0.6, observed_latency_ms / active_base_lat)

        excess_load = max(0.0, current_workload_req_per_min - 60.0)
        if excess_load > 0:
            projected_lat = base_lat + excess_load * lat_sens
            projected_cpu = base_cpu + excess_load * cpu_sens
        else:
            # Scale candidate latency proportionally by the stress ratio softened by model sensitivity
            sensitivity_ratio = lat_sens / max(0.05, float(active_profile.get("latency_load_sensitivity", 0.3)))
            scaled_factor = 1.0 + (stress_ratio - 1.0) * min(1.15, max(0.35, sensitivity_ratio))
            projected_lat = max(5.0, base_lat * scaled_factor)
            projected_cpu = min(98.0, max(15.0, base_cpu * (1.0 + (stress_ratio - 1.0) * 0.45)))

        acc = float(candidate_profile["accuracy"])
        lat_ok = projected_lat <= self.max_latency_sla_ms
        cpu_ok = projected_cpu <= self.max_cpu_sla_percent
        acc_ok = acc >= self.min_accuracy_sla

        return {
            "model_id": candidate_profile["id"],
            "name": candidate_profile["name"],
            "tier": candidate_profile["tier"],
            "accuracy": acc,
            "projected_latency_ms": round(projected_lat, 1),
            "projected_cpu_percent": round(projected_cpu, 1),
            "satisfies_accuracy_sla": acc_ok,
            "satisfies_latency_sla": lat_ok,
            "satisfies_cpu_sla": cpu_ok,
            "is_feasible": acc_ok and lat_ok and cpu_ok,
        }


sla_manager = SLAManagerModule()
