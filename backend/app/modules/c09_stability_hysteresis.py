"""
Component 9: Stability & Hysteresis Manager
===========================================
Responsibilities:
- Prevents model oscillation / thrashing in both directions:
  1. Downward Adaptation (SLA Violation):
     Distinguishes temporary fluctuations (e.g., 90 -> 105 -> 98 -> 103 -> 95 ms)
     from persistent SLA violations (e.g., 105 -> 112 -> 120 -> 130 -> 125 ms) using a
     consecutive violation counter (`persistent_violation_window`).
  2. Upward Adaptation (Cooldown & Hysteresis Recovery):
     When workload cools down, waits for:
     - Cooldown period (`cooldown_seconds`) after the last switch,
     - Sustained stability window (`recovery_stability_window` consecutive compliant samples), AND
     - Hysteresis safety margin: the heavier, higher-accuracy model's projected latency
       must be safely below `SLA - hysteresis_latency_margin_ms` before switching back.
"""
import time
from typing import Dict, Any, Optional
from backend.app.config import config


class StabilityHysteresisManager:
    """Manages persistence windows, cooldown timers, and hysteresis margins to prevent thrashing."""

    def __init__(self):
        self.persistent_violation_window: int = config.persistent_violation_window
        self.recovery_stability_window: int = config.recovery_stability_window
        self.cooldown_seconds: float = config.cooldown_seconds
        self.hysteresis_latency_margin_ms: float = config.hysteresis_latency_margin_ms

        self.consecutive_violations: int = 0
        self.consecutive_stable_samples: int = 0
        self.transient_fluctuations_absorbed: int = 0
        self.thrashing_prevented_count: int = 0
        self.last_adaptation_time: float = 0.0
        self.last_decision_state: str = "STABLE_MONITORING"
        self.last_decision_explanation: str = "System operating normally within SLA bounds."

    def update_config(
        self,
        persistent_violation_window: Optional[int] = None,
        recovery_stability_window: Optional[int] = None,
        cooldown_seconds: Optional[float] = None,
        hysteresis_latency_margin_ms: Optional[float] = None,
    ) -> Dict[str, Any]:
        if persistent_violation_window is not None:
            self.persistent_violation_window = int(persistent_violation_window)
        if recovery_stability_window is not None:
            self.recovery_stability_window = int(recovery_stability_window)
        if cooldown_seconds is not None:
            self.cooldown_seconds = float(cooldown_seconds)
        if hysteresis_latency_margin_ms is not None:
            self.hysteresis_latency_margin_ms = float(hysteresis_latency_margin_ms)
        return self.get_status()

    def reset_counters(self):
        self.consecutive_violations = 0
        self.consecutive_stable_samples = 0
        self.last_adaptation_time = 0.0
        self.last_decision_state = "STABLE_MONITORING"
        self.last_decision_explanation = "Stability counters reset."

    def record_switch_executed(self):
        self.last_adaptation_time = time.time()
        self.consecutive_violations = 0
        self.consecutive_stable_samples = 0

    def evaluate_stability(
        self,
        sla_satisfied: bool,
        active_model_id: str,
        highest_accuracy_model_id: str,
        projected_upgrade_latency_ms: float,
        max_latency_sla_ms: float,
        ignore_cooldown: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluates current SLA compliance sample through the persistence & hysteresis filter.
        Returns a structured decision dict telling the MAPE-K engine whether to trigger adaptation.
        """
        now = time.time()
        elapsed_since_switch = now - self.last_adaptation_time if self.last_adaptation_time > 0 else 9999.0
        in_cooldown = (not ignore_cooldown) and (elapsed_since_switch < self.cooldown_seconds)
        cooldown_remaining = max(0.0, round(self.cooldown_seconds - elapsed_since_switch, 1)) if in_cooldown else 0.0

        # Case A: Current sample violates SLA
        if not sla_satisfied:
            self.consecutive_violations += 1
            self.consecutive_stable_samples = 0

            if in_cooldown:
                self.thrashing_prevented_count += 1
                self.last_decision_state = "COOLDOWN_HOLD"
                self.last_decision_explanation = (
                    f"SLA violation observed ({self.consecutive_violations}/{self.persistent_violation_window}), "
                    f"but system is in post-switch cooldown ({cooldown_remaining}s remaining)."
                )
                return {
                    "trigger_adaptation": False,
                    "direction": None,
                    "state": self.last_decision_state,
                    "explanation": self.last_decision_explanation,
                    "consecutive_violations": self.consecutive_violations,
                    "consecutive_stable_samples": self.consecutive_stable_samples,
                    "cooldown_remaining_sec": cooldown_remaining,
                }

            if self.consecutive_violations < self.persistent_violation_window:
                self.last_decision_state = "TRANSIENT_WATCH"
                self.last_decision_explanation = (
                    f"Temporary SLA breach ({self.consecutive_violations}/{self.persistent_violation_window} consecutive). "
                    f"Holding current model to filter out transient fluctuation."
                )
                return {
                    "trigger_adaptation": False,
                    "direction": None,
                    "state": self.last_decision_state,
                    "explanation": self.last_decision_explanation,
                    "consecutive_violations": self.consecutive_violations,
                    "consecutive_stable_samples": self.consecutive_stable_samples,
                    "cooldown_remaining_sec": 0.0,
                }

            # Persistent violation confirmed!
            self.last_decision_state = "PERSISTENT_VIOLATION_CONFIRMED"
            self.last_decision_explanation = (
                f"Persistent SLA violation confirmed ({self.consecutive_violations}/{self.persistent_violation_window} "
                f"consecutive breaches). Triggering MAPE-K model adaptation."
            )
            return {
                "trigger_adaptation": True,
                "direction": "DOWNGRADE_FOR_SLA",
                "state": self.last_decision_state,
                "explanation": self.last_decision_explanation,
                "consecutive_violations": self.consecutive_violations,
                "consecutive_stable_samples": self.consecutive_stable_samples,
                "cooldown_remaining_sec": 0.0,
            }

        # Case B: Current sample satisfies SLA
        else:
            if self.consecutive_violations > 0:
                # A prior brief spike just recovered before hitting the persistent threshold!
                self.transient_fluctuations_absorbed += 1
                prev_v = self.consecutive_violations
                self.consecutive_violations = 0
                self.consecutive_stable_samples = 1
                self.last_decision_state = "TRANSIENT_FLUCTUATION_ABSORBED"
                self.last_decision_explanation = (
                    f"Latency returned within SLA after {prev_v} brief breach(es). "
                    f"Classified as temporary fluctuation — unnecessary model switch prevented!"
                )
            else:
                self.consecutive_stable_samples += 1

            # Check if we are running a lower-accuracy model and could upgrade back to higher accuracy
            if active_model_id != highest_accuracy_model_id:
                hysteresis_ceiling = max_latency_sla_ms - self.hysteresis_latency_margin_ms
                hysteresis_safe = projected_upgrade_latency_ms <= hysteresis_ceiling

                if in_cooldown:
                    self.thrashing_prevented_count += 1
                    self.last_decision_state = "RECOVERY_COOLDOWN_WAIT"
                    self.last_decision_explanation = (
                        f"Workload is calm, but holding {active_model_id} during cooldown "
                        f"({cooldown_remaining}s remaining) to prevent oscillation."
                    )
                    return {
                        "trigger_adaptation": False,
                        "direction": None,
                        "state": self.last_decision_state,
                        "explanation": self.last_decision_explanation,
                        "consecutive_violations": 0,
                        "consecutive_stable_samples": self.consecutive_stable_samples,
                        "cooldown_remaining_sec": cooldown_remaining,
                    }

                if self.consecutive_stable_samples < self.recovery_stability_window:
                    self.last_decision_state = "RECOVERY_STABILIZING"
                    self.last_decision_explanation = (
                        f"Conditions normal ({self.consecutive_stable_samples}/{self.recovery_stability_window} "
                        f"stable samples). Waiting for stability window before switching back."
                    )
                    return {
                        "trigger_adaptation": False,
                        "direction": None,
                        "state": self.last_decision_state,
                        "explanation": self.last_decision_explanation,
                        "consecutive_violations": 0,
                        "consecutive_stable_samples": self.consecutive_stable_samples,
                        "cooldown_remaining_sec": 0.0,
                    }

                if not hysteresis_safe:
                    self.thrashing_prevented_count += 1
                    self.last_decision_state = "HYSTERESIS_MARGIN_BLOCK"
                    self.last_decision_explanation = (
                        f"Stability window met, but upgrading to {highest_accuracy_model_id} projects "
                        f"{projected_upgrade_latency_ms:.1f}ms latency (exceeds hysteresis safe ceiling of "
                        f"{hysteresis_ceiling:.1f}ms). Holding {active_model_id} to prevent thrashing."
                    )
                    return {
                        "trigger_adaptation": False,
                        "direction": None,
                        "state": self.last_decision_state,
                        "explanation": self.last_decision_explanation,
                        "consecutive_violations": 0,
                        "consecutive_stable_samples": self.consecutive_stable_samples,
                        "cooldown_remaining_sec": 0.0,
                    }

                # Both stability window and hysteresis margin satisfied -> Upgrade back!
                self.last_decision_state = "BIDIRECTIONAL_RECOVERY_READY"
                self.last_decision_explanation = (
                    f"Conditions stable for {self.consecutive_stable_samples}/{self.recovery_stability_window} samples "
                    f"and projected {highest_accuracy_model_id} latency ({projected_upgrade_latency_ms:.1f}ms) is safely "
                    f"below hysteresis threshold ({hysteresis_ceiling:.1f}ms). Switching back to optimal accuracy model."
                )
                return {
                    "trigger_adaptation": True,
                    "direction": "UPGRADE_FOR_ACCURACY",
                    "state": self.last_decision_state,
                    "explanation": self.last_decision_explanation,
                    "consecutive_violations": 0,
                    "consecutive_stable_samples": self.consecutive_stable_samples,
                    "cooldown_remaining_sec": 0.0,
                }

            if self.last_decision_state != "TRANSIENT_FLUCTUATION_ABSORBED":
                self.last_decision_state = "OPTIMAL_STABLE"
                self.last_decision_explanation = (
                    f"Optimal model ({active_model_id}) active and satisfying all SLA constraints."
                )

            return {
                "trigger_adaptation": False,
                "direction": None,
                "state": self.last_decision_state,
                "explanation": self.last_decision_explanation,
                "consecutive_violations": 0,
                "consecutive_stable_samples": self.consecutive_stable_samples,
                "cooldown_remaining_sec": cooldown_remaining,
            }

    def get_status(self) -> Dict[str, Any]:
        now = time.time()
        elapsed = now - self.last_adaptation_time if self.last_adaptation_time > 0 else 9999.0
        cooldown_rem = max(0.0, round(self.cooldown_seconds - elapsed, 1)) if self.last_adaptation_time > 0 else 0.0
        return {
            "persistent_violation_window": self.persistent_violation_window,
            "recovery_stability_window": self.recovery_stability_window,
            "cooldown_seconds": self.cooldown_seconds,
            "hysteresis_latency_margin_ms": self.hysteresis_latency_margin_ms,
            "consecutive_violations": self.consecutive_violations,
            "consecutive_stable_samples": self.consecutive_stable_samples,
            "transient_fluctuations_absorbed": self.transient_fluctuations_absorbed,
            "thrashing_prevented_count": self.thrashing_prevented_count,
            "cooldown_remaining_sec": cooldown_rem,
            "state": self.last_decision_state,
            "explanation": self.last_decision_explanation,
        }


stability_manager = StabilityHysteresisManager()
