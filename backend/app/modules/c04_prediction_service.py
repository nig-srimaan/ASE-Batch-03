"""
Component 4: Prediction Service (with Network Flow Presets & Feature Explainability)
====================================================================================
Responsibilities:
- Serves real-time inference requests using whichever ML model is currently active
- Provides curated Cybersecurity Network Traffic Presets (Normal HTTPS, SYN Flood DDoS,
  Port Scan Probe, Exfiltration Burst) for live interactive testing
- Computes per-prediction Top-5 Feature Contributions (local sensitivity attribution)
- Records runtime latency, CPU, and prediction confidence into the Runtime Monitoring Module
"""
import numpy as np
from typing import List, Optional, Dict, Any

from backend.app.modules.c01_data_management import data_manager, FEATURE_NAMES
from backend.app.modules.c05_runtime_monitoring import runtime_monitor
from backend.app.modules.c06_sla_manager import sla_manager
from backend.app.modules.c10_model_switching import switching_manager


TRAFFIC_PRESETS: Dict[str, Dict[str, Any]] = {
    "normal_web_https": {
        "name": "Normal HTTPS Web Browsing Flow",
        "description": "Balanced bidirectional TLS session with normal packet inter-arrival times.",
        "features": [-0.8, 0.7, -0.6, 0.5, -0.5, 0.4, 0.1, -0.3, 0.3, -0.4, 0.5, -0.2, -0.3, 0.1, -0.2, 0.2, 0.1, -0.1, 0.2, -0.2],
    },
    "syn_flood_ddos": {
        "name": "SYN Flood DDoS Attack Burst",
        "description": "High-rate half-open SYN flood with extreme packet rate and anomalous flow flags.",
        "features": [1.8, 1.4, 1.5, -1.3, 1.2, -1.1, 0.2, 1.2, -0.9, 1.5, 1.1, -0.8, 1.2, 1.3, 1.4, 0.9, -0.7, 1.1, -0.5, 0.8],
    },
    "port_scan_probe": {
        "name": "Stealth SYN Port Scan Probe",
        "description": "Rapid short-duration probing across sequential ports with minimal payload bytes.",
        "features": [1.3, -1.1, 1.2, 1.1, 0.8, -0.9, 0.1, 0.9, -0.6, 1.2, 0.8, -0.5, 0.9, 1.1, 0.9, -0.4, 0.6, 0.5, -0.3, 0.4],
    },
    "benign_file_transfer": {
        "name": "Benign Cloud Backup Transfer",
        "description": "Sustained high-byte legitimate backup stream with steady ACK ratios.",
        "features": [-1.2, 0.9, -0.9, 0.8, -0.7, 0.7, 1.4, -0.5, 0.4, -0.8, -0.6, 0.7, -0.4, 0.3, -0.5, 0.4, 0.3, -0.2, 0.1, -0.3],
    },
}


class PredictionServiceModule:
    """Runtime prediction service routing requests to the active ML model with feature attribution."""

    @staticmethod
    def _compute_top_feature_attributions(model: Any, x_arr: np.ndarray, base_prob: float) -> List[Dict[str, Any]]:
        """Computes local perturbation sensitivity to identify the top 5 driving features."""
        attributions = []
        for idx in range(min(len(FEATURE_NAMES), x_arr.shape[1])):
            val = float(x_arr[0, idx])
            x_pert = x_arr.copy()
            x_pert[0, idx] = 0.0
            try:
                pert_prob = float(model.predict_proba(x_pert)[0][1])
                impact = base_prob - pert_prob
            except Exception:
                impact = val * 0.05
            attributions.append({
                "feature": FEATURE_NAMES[idx],
                "value": round(val, 3),
                "impact_on_attack_prob": round(float(impact), 4),
                "direction": "INCREASES_RISK" if impact >= 0 else "DECREASES_RISK",
            })
        attributions.sort(key=lambda d: abs(d["impact_on_attack_prob"]), reverse=True)
        return attributions[:5]

    def predict(
        self,
        features: Optional[List[float]] = None,
        workload_multiplier: Optional[float] = None,
        induced_latency_ms: Optional[float] = None,
        preset_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        model_id, model, profile = switching_manager.get_active_model_bundle()

        preset_used = None
        if preset_id and preset_id in TRAFFIC_PRESETS:
            features = TRAFFIC_PRESETS[preset_id]["features"]
            preset_used = TRAFFIC_PRESETS[preset_id]["name"]
        elif not features or len(features) != data_manager.n_features:
            features = data_manager.sample_feature_vector()

        x_arr = np.array(features, dtype=np.float64).reshape(1, -1)
        probs = model.predict_proba(x_arr)[0]
        pred_class = int(probs[1] >= 0.5)
        confidence = float(max(probs[0], probs[1]))
        top_features = self._compute_top_feature_attributions(model, x_arr, float(probs[1]))

        workload_override = None
        if workload_multiplier is not None:
            workload_override = 60.0 * float(workload_multiplier)

        latency_ms, cpu_pct, mem_pct = runtime_monitor.compute_realistic_telemetry(
            model_profile=profile,
            explicit_latency_ms=induced_latency_ms,
            workload_override=workload_override,
        )

        runtime_monitor.record_observation(
            latency_ms=latency_ms,
            cpu_percent=cpu_pct,
            memory_percent=mem_pct,
            confidence=confidence,
        )

        sla_satisfied = (
            latency_ms <= sla_manager.max_latency_sla_ms
            and cpu_pct <= sla_manager.max_cpu_sla_percent
            and float(profile["accuracy"]) >= sla_manager.min_accuracy_sla
        )

        return {
            "prediction": pred_class,
            "label": "INTRUSION_ATTACK" if pred_class == 1 else "NORMAL_TRAFFIC",
            "confidence": round(confidence, 4),
            "probabilities": [round(float(probs[0]), 4), round(float(probs[1]), 4)],
            "served_by_model": model_id,
            "model_display_name": profile["name"],
            "model_accuracy": float(profile["accuracy"]),
            "latency_ms": latency_ms,
            "cpu_percent": cpu_pct,
            "workload_req_per_min": round(runtime_monitor.workload_req_per_min, 1),
            "sla_satisfied": sla_satisfied,
            "active_sla_target_ms": sla_manager.max_latency_sla_ms,
            "preset_used": preset_used,
            "top_feature_attributions": top_features,
        }


prediction_service = PredictionServiceModule()
