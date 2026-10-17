"""
Component 8: Adaptive Model Selector
====================================
Responsibilities:
- Evaluates all candidate ML models when an adaptation is triggered
- Filters out candidate models that violate hard SLA constraints
  (e.g., rejecting Logistic Regression at 82% accuracy when Accuracy SLA >= 85%)
- Ranks feasible candidate models by multi-criteria utility:
  * Primary objective: Maximize model accuracy
  * Secondary objective: Latency & CPU headroom under current workload
"""
from typing import Dict, Any, List, Optional, Tuple
from backend.app.modules.c06_sla_manager import sla_manager


class AdaptiveModelSelector:
    """Multi-criteria SLA-constrained model selection engine."""

    def select_best_model(
        self,
        active_model_id: str,
        all_profiles: Dict[str, Dict[str, Any]],
        current_workload_req_per_min: float,
        observed_latency_ms: float,
        reason_mode: str = "SLA_VIOLATION",
    ) -> Tuple[Optional[str], List[Dict[str, Any]], str]:
        """
        Evaluates all models in `all_profiles` and returns:
        `(selected_model_id, candidate_evaluations, selection_rationale)`
        """
        active_profile = all_profiles[active_model_id]
        evaluations: List[Dict[str, Any]] = []

        for model_id, profile in all_profiles.items():
            proj = sla_manager.project_candidate_performance(
                candidate_profile=profile,
                current_workload_req_per_min=current_workload_req_per_min,
                active_profile=active_profile,
                observed_latency_ms=observed_latency_ms,
            )

            # Calculate utility score for feasible candidates:
            # High weight on accuracy (70%), plus latency headroom (20%) & CPU headroom (10%)
            lat_headroom = max(0.0, (sla_manager.max_latency_sla_ms - proj["projected_latency_ms"]) / sla_manager.max_latency_sla_ms)
            cpu_headroom = max(0.0, (sla_manager.max_cpu_sla_percent - proj["projected_cpu_percent"]) / sla_manager.max_cpu_sla_percent)
            utility_score = round(0.70 * proj["accuracy"] + 0.20 * lat_headroom + 0.10 * cpu_headroom, 4)

            rejection_reasons = []
            if not proj["satisfies_accuracy_sla"]:
                rejection_reasons.append(
                    f"Accuracy {proj['accuracy']*100:.0f}% < SLA minimum {sla_manager.min_accuracy_sla*100:.0f}%"
                )
            if not proj["satisfies_latency_sla"]:
                rejection_reasons.append(
                    f"Projected latency {proj['projected_latency_ms']:.1f}ms > SLA limit {sla_manager.max_latency_sla_ms:.0f}ms"
                )
            if not proj["satisfies_cpu_sla"]:
                rejection_reasons.append(
                    f"Projected CPU {proj['projected_cpu_percent']:.1f}% > SLA limit {sla_manager.max_cpu_sla_percent:.0f}%"
                )

            evaluations.append({
                **proj,
                "utility_score": utility_score,
                "is_current_model": model_id == active_model_id,
                "rejection_reasons": rejection_reasons,
                "status": "FEASIBLE" if proj["is_feasible"] else ("REJECTED: " + "; ".join(rejection_reasons)),
            })

        # Sort evaluations by accuracy descending, then projected latency ascending
        evaluations.sort(key=lambda x: (-x["accuracy"], x["projected_latency_ms"]))

        feasible_candidates = [e for e in evaluations if e["is_feasible"]]

        if feasible_candidates:
            # Prefer highest accuracy among feasible models (and lower latency on tie)
            best = max(feasible_candidates, key=lambda x: (x["accuracy"], x["utility_score"]))
            rationale = (
                f"Selected {best['name']} (Accuracy {best['accuracy']*100:.0f}%, "
                f"Projected Latency {best['projected_latency_ms']:.1f}ms <= {sla_manager.max_latency_sla_ms:.0f}ms SLA). "
            )
            rejected_notes = [
                f"{e['name']} rejected ({'; '.join(e['rejection_reasons'])})"
                for e in evaluations if not e["is_feasible"]
            ]
            if rejected_notes:
                rationale += " | ".join(rejected_notes)
            return best["model_id"], evaluations, rationale

        # If no candidate satisfies all hard SLAs:
        if sla_manager.allow_emergency_degrade_below_accuracy_sla:
            # Emergency survival mode: choose model that satisfies latency SLA with highest accuracy
            lat_feasible = [e for e in evaluations if e["satisfies_latency_sla"]]
            if lat_feasible:
                best_emerg = max(lat_feasible, key=lambda x: x["accuracy"])
                return (
                    best_emerg["model_id"],
                    evaluations,
                    f"Emergency fallback to {best_emerg['name']}: no model met strict Accuracy+Latency SLA simultaneously.",
                )

        return (
            None,
            evaluations,
            "No alternative model satisfies all hard SLA constraints (Accuracy floor protected).",
        )


model_selector = AdaptiveModelSelector()
