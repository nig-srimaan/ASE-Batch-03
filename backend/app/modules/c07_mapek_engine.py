"""
Component 7: MAPE-K Adaptation Engine
=====================================
Responsibilities:
- Orchestrates the closed-loop self-adaptive feedback cycle:
  1. MONITOR:   Collects latest runtime telemetry from Runtime Monitoring Module
  2. ANALYZE:   Checks SLA compliance (SLA Manager) and filters transient spikes vs.
                persistent violations / hysteresis stability (Stability & Hysteresis Manager)
  3. PLAN:      Invokes Adaptive Model Selector to evaluate alternatives against hard SLA
                constraints and rank feasible candidates by accuracy & utility
  4. EXECUTE:   Invokes Model Switching Manager to hot-swap the active model when needed
  5. KNOWLEDGE: Persists telemetry snapshots, candidate evaluations, and adaptation events
                in the Knowledge Base
"""
import time
from typing import Dict, Any, Optional

from backend.app.modules.c03_model_evaluation import model_evaluator
from backend.app.modules.c05_runtime_monitoring import runtime_monitor
from backend.app.modules.c06_sla_manager import sla_manager
from backend.app.modules.c08_model_selector import model_selector
from backend.app.modules.c09_stability_hysteresis import stability_manager
from backend.app.modules.c10_model_switching import switching_manager
from backend.app.modules.c11_knowledge_base import knowledge_base


class MAPEKAdaptationEngine:
    """Core self-adaptive software engineering engine implementing the MAPE-K loop."""

    def __init__(self):
        self.cycle_count: int = 0
        self.last_cycle_trace: Dict[str, Any] = {}

    def execute_cycle(self, ignore_cooldown: bool = False) -> Dict[str, Any]:
        """Runs one complete Monitor -> Analyze -> Plan -> Execute -> Knowledge iteration."""
        self.cycle_count += 1
        now = time.time()

        # ---------------------------------------------------------------------
        # 1. MONITOR PHASE
        # ---------------------------------------------------------------------
        active_model_id, _, active_profile = switching_manager.get_active_model_bundle()
        all_profiles = model_evaluator.get_profiles()
        metrics = runtime_monitor.get_metrics_snapshot()

        monitor_phase = {
            "active_model_id": active_model_id,
            "active_model_name": active_profile["name"],
            "latest_latency_ms": metrics["latest_latency_ms"],
            "p95_latency_ms": metrics["p95_latency_ms"],
            "cpu_percent": metrics["cpu_percent"],
            "memory_percent": metrics["memory_percent"],
            "workload_req_per_min": metrics["workload_req_per_min"],
        }

        # ---------------------------------------------------------------------
        # 2. ANALYZE PHASE (SLA Check + Stability/Hysteresis Evaluation)
        # ---------------------------------------------------------------------
        sla_eval = sla_manager.evaluate_current_compliance(active_profile, metrics)

        # Project what the highest-accuracy model (xgboost) would look like for recovery analysis
        xgb_proj = sla_manager.project_candidate_performance(
            candidate_profile=all_profiles["xgboost"],
            current_workload_req_per_min=metrics["workload_req_per_min"],
            active_profile=active_profile,
            observed_latency_ms=metrics["latest_latency_ms"],
        )

        stability_eval = stability_manager.evaluate_stability(
            sla_satisfied=sla_eval["sla_satisfied"],
            active_model_id=active_model_id,
            highest_accuracy_model_id="xgboost",
            projected_upgrade_latency_ms=xgb_proj["projected_latency_ms"],
            max_latency_sla_ms=sla_manager.max_latency_sla_ms,
            ignore_cooldown=ignore_cooldown,
        )

        analyze_phase = {
            "sla_satisfied": sla_eval["sla_satisfied"],
            "violations": sla_eval["violations"],
            "stability_state": stability_eval["state"],
            "stability_explanation": stability_eval["explanation"],
            "consecutive_violations": stability_eval["consecutive_violations"],
            "consecutive_stable_samples": stability_eval["consecutive_stable_samples"],
            "trigger_adaptation": stability_eval["trigger_adaptation"],
            "direction": stability_eval["direction"],
        }

        # ---------------------------------------------------------------------
        # 3. PLAN PHASE (Adaptive Model Selection)
        # ---------------------------------------------------------------------
        selected_model_id, candidate_evaluations, selection_rationale = model_selector.select_best_model(
            active_model_id=active_model_id,
            all_profiles=all_profiles,
            current_workload_req_per_min=metrics["workload_req_per_min"],
            observed_latency_ms=metrics["latest_latency_ms"],
            reason_mode=stability_eval["direction"] or "MONITOR_ONLY",
        )

        should_switch = (
            stability_eval["trigger_adaptation"]
            and selected_model_id is not None
            and selected_model_id != active_model_id
        )

        plan_phase = {
            "adaptation_triggered": stability_eval["trigger_adaptation"],
            "recommended_model_id": selected_model_id,
            "should_execute_switch": should_switch,
            "selection_rationale": selection_rationale,
            "candidate_evaluations": candidate_evaluations,
        }

        # ---------------------------------------------------------------------
        # 4. EXECUTE PHASE (Zero-Downtime Model Switching)
        # ---------------------------------------------------------------------
        adaptation_event: Optional[Dict[str, Any]] = None
        if should_switch:
            reason_label = (
                f"Persistent latency/SLA violation ({metrics['latest_latency_ms']:.1f}ms > {sla_manager.max_latency_sla_ms:.0f}ms)"
                if stability_eval["direction"] == "DOWNGRADE_FOR_SLA"
                else f"Bidirectional recovery after sustained stability & hysteresis verification ({metrics['latest_latency_ms']:.1f}ms)"
            )
            exec_result = switching_manager.execute_switch(selected_model_id, reason=reason_label)
            if exec_result.get("switched"):
                stability_manager.record_switch_executed()
                # Update runtime monitor baseline observation to reflect the new model's latency under current load
                new_proj = next(
                    (c for c in candidate_evaluations if c["model_id"] == selected_model_id),
                    None
                )
                if new_proj:
                    runtime_monitor.record_observation(
                        latency_ms=new_proj["projected_latency_ms"],
                        cpu_percent=new_proj["projected_cpu_percent"],
                        memory_percent=metrics["memory_percent"],
                    )
                execute_phase = {
                    "action": "SWITCHED_ACTIVE_MODEL",
                    "from_model": active_model_id,
                    "to_model": selected_model_id,
                    "warmup_verification_ms": exec_result["warmup_verification_ms"],
                    "reason": reason_label,
                }
            else:
                execute_phase = {
                    "action": "SWITCH_ABORTED",
                    "message": exec_result.get("message", "Switch not executed"),
                }
        else:
            execute_phase = {
                "action": "NO_OP_MAINTAIN_MODEL",
                "active_model": active_model_id,
                "reason": stability_eval["explanation"] if not stability_eval["trigger_adaptation"] else selection_rationale,
            }

        # ---------------------------------------------------------------------
        # 5. KNOWLEDGE PHASE (Persist Telemetry & Adaptation Event)
        # ---------------------------------------------------------------------
        current_active_after = switching_manager.active_model_id
        knowledge_base.record_telemetry_snapshot({
            "timestamp": now,
            "active_model": current_active_after,
            "latency_ms": metrics["latest_latency_ms"],
            "p95_latency_ms": metrics["p95_latency_ms"],
            "cpu_percent": metrics["cpu_percent"],
            "memory_percent": metrics["memory_percent"],
            "workload_req_per_min": metrics["workload_req_per_min"],
            "sla_satisfied": sla_eval["sla_satisfied"],
            "consecutive_violations": stability_manager.consecutive_violations,
            "consecutive_stable": stability_manager.consecutive_stable_samples,
            "event_tag": execute_phase["action"],
        })

        full_trace = {
            "cycle_number": self.cycle_count,
            "timestamp": now,
            "iso_time": time.strftime("%H:%M:%S", time.localtime(now)),
            "monitor": monitor_phase,
            "analyze": analyze_phase,
            "plan": plan_phase,
            "execute": execute_phase,
        }

        if execute_phase["action"] == "SWITCHED_ACTIVE_MODEL":
            adaptation_event = knowledge_base.record_adaptation({
                "timestamp": now,
                "from_model": active_model_id,
                "to_model": selected_model_id,
                "adaptation_type": stability_eval["direction"],
                "reason": execute_phase["reason"],
                "trigger_latency_ms": metrics["latest_latency_ms"],
                "trigger_cpu_percent": metrics["cpu_percent"],
                "workload_req_per_min": metrics["workload_req_per_min"],
                "sla_latency_ms": sla_manager.max_latency_sla_ms,
                "sla_min_accuracy": sla_manager.min_accuracy_sla,
                "candidate_evaluations": candidate_evaluations,
                "mapek_trace": full_trace,
            })

        full_trace["adaptation_logged"] = adaptation_event
        self.last_cycle_trace = full_trace
        return full_trace


mapek_engine = MAPEKAdaptationEngine()
