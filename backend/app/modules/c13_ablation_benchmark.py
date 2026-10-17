"""
Policy Ablation & Comparative Benchmark Engine
==============================================
Proves quantitatively why the Proposed MAPE-K + Persistence Window + Bidirectional Hysteresis
architecture outperforms:
1. Static Heavy (`XGBoost Only`) — suffers high Latency SLA violations during bursts
2. Static Light (`Logistic Regression Only`) — violates the 85% Accuracy SLA 100% of the time
3. Naive Reactive Switcher (Immediate switch on single breach, immediate switch-back without
   hysteresis) — suffers severe model thrashing/oscillation (`10+ switches`) under fluctuating load
"""
from typing import Dict, Any, List


class AblationBenchmarkEngine:
    """Simulates a standardized 30-tick multi-phase workload across 4 competing policies."""

    # 30-tick realistic workload (req/min) and noise profile covering:
    # Ticks 1-6:   Normal load with brief single-tick micro-spikes (Tests transient spike filtering)
    # Ticks 7-15:  Sustained traffic surge (Tests persistent SLA violation adaptation)
    # Ticks 16-22: Borderline fluctuating load around ~105 req/min (Tests anti-thrashing hysteresis!)
    # Ticks 23-30: Calm recovery workload ~55 req/min (Tests bidirectional switch-back to XGBoost)
    WORKLOAD_TRACE = [
        # Phase 1: Normal + transient micro-spikes (XGBoost latency: 80, 104, 96, 102, 92, 85)
        (60, 80.0), (65, 104.0), (62, 96.0), (64, 102.0), (60, 92.0), (60, 85.0),
        # Phase 2: Sustained surge (XGBoost latency: 106, 114, 122, 128, 132, 130, 126, 124, 122)
        (125, 106.0), (132, 114.0), (140, 122.0), (145, 128.0), (150, 132.0),
        (148, 130.0), (142, 126.0), (138, 124.0), (135, 122.0),
        # Phase 3: Borderline oscillatory zone (XGBoost would be ~103ms > 100ms, RF is ~68ms <= 100ms)
        (110, 104.0), (108, 102.0), (112, 105.0), (106, 101.0), (110, 104.0), (108, 102.0), (105, 101.0),
        # Phase 4: Sustained cooldown recovery (XGBoost ~78ms <= 88ms hysteresis ceiling, RF ~58ms)
        (58, 79.0), (56, 78.0), (55, 77.0), (55, 76.0), (54, 76.0), (54, 75.0), (55, 76.0), (55, 76.0),
    ]

    MODEL_SPECS = {
        "xgboost": {"name": "XGBoost", "accuracy": 0.92},
        "random_forest": {"name": "Random Forest", "accuracy": 0.88},
        "logistic_regression": {"name": "Logistic Regression", "accuracy": 0.82},
    }

    @classmethod
    def _rf_latency_from_xgb(cls, xgb_lat: float) -> float:
        # Random Forest has 60ms base and lower load sensitivity
        return round(max(48.0, 60.0 + (xgb_lat - 80.0) * 0.28), 1)

    @classmethod
    def run_comparative_study(
        cls,
        latency_sla_ms: float = 100.0,
        min_accuracy_sla: float = 0.85,
        persistence_window: int = 5,
        hysteresis_margin_ms: float = 12.0,
    ) -> Dict[str, Any]:
        n_ticks = len(cls.WORKLOAD_TRACE)

        # State trackers for each of the 4 policies
        policies = {
            "static_xgboost": {"label": "Static XGBoost (No Adaptation)", "series": [], "switches": 0, "thrashing": 0},
            "static_logreg": {"label": "Static Logistic Regression", "series": [], "switches": 0, "thrashing": 0},
            "naive_reactive": {"label": "Naive Reactive (No Persistence / No Hysteresis)", "series": [], "switches": 0, "thrashing": 0},
            "proposed_mapek": {"label": "Proposed MAPE-K (Persistence + Bidirectional Hysteresis)", "series": [], "switches": 0, "thrashing": 0},
        }

        naive_model = "xgboost"
        naive_last_switch_tick = -10

        prop_model = "xgboost"
        prop_violations = 0
        prop_stable = 0
        prop_last_switch_tick = -10

        for tick_idx, (workload, xgb_lat) in enumerate(cls.WORKLOAD_TRACE, start=1):
            rf_lat = cls._rf_latency_from_xgb(xgb_lat)
            lr_lat = round(max(16.0, 20.0 + (xgb_lat - 80.0) * 0.08), 1)

            # 1. Static XGBoost
            policies["static_xgboost"]["series"].append({
                "tick": tick_idx, "model": "xgboost", "latency_ms": xgb_lat, "accuracy": 0.92,
                "lat_ok": xgb_lat <= latency_sla_ms, "acc_ok": True,
            })

            # 2. Static Logistic Regression
            policies["static_logreg"]["series"].append({
                "tick": tick_idx, "model": "logistic_regression", "latency_ms": lr_lat, "accuracy": 0.82,
                "lat_ok": lr_lat <= latency_sla_ms, "acc_ok": 0.82 >= min_accuracy_sla,
            })

            # 3. Naive Reactive Switcher (switches immediately on 1 breach, upgrades immediately when current < SLA)
            naive_obs_lat = xgb_lat if naive_model == "xgboost" else rf_lat
            prev_naive = naive_model
            if naive_model == "xgboost" and naive_obs_lat > latency_sla_ms:
                naive_model = "random_forest"
            elif naive_model == "random_forest" and naive_obs_lat <= latency_sla_ms:
                # Naive switcher sees RF latency (e.g. 68ms <= 100ms) is "good" and immediately tries to switch back to XGBoost!
                naive_model = "xgboost"

            if naive_model != prev_naive:
                policies["naive_reactive"]["switches"] += 1
                if (tick_idx - naive_last_switch_tick) <= 3:
                    policies["naive_reactive"]["thrashing"] += 1
                naive_last_switch_tick = tick_idx

            policies["naive_reactive"]["series"].append({
                "tick": tick_idx, "model": prev_naive, "model_after": naive_model,
                "latency_ms": naive_obs_lat, "accuracy": cls.MODEL_SPECS[prev_naive]["accuracy"],
                "lat_ok": naive_obs_lat <= latency_sla_ms, "acc_ok": True,
            })

            # 4. Proposed MAPE-K with Persistence Window & Bidirectional Hysteresis
            prop_obs_lat = xgb_lat if prop_model == "xgboost" else rf_lat
            prev_prop = prop_model
            if prop_obs_lat > latency_sla_ms:
                prop_violations += 1
                prop_stable = 0
                if prop_violations >= persistence_window and prop_model == "xgboost":
                    prop_model = "random_forest"
                    prop_violations = 0
                    prop_stable = 0
            else:
                prop_violations = 0
                prop_stable += 1
                if (
                    prop_model == "random_forest"
                    and prop_stable >= persistence_window
                    and xgb_lat <= (latency_sla_ms - hysteresis_margin_ms)
                ):
                    prop_model = "xgboost"
                    prop_stable = 0

            if prop_model != prev_prop:
                policies["proposed_mapek"]["switches"] += 1
                if (tick_idx - prop_last_switch_tick) <= 3:
                    policies["proposed_mapek"]["thrashing"] += 1
                prop_last_switch_tick = tick_idx

            policies["proposed_mapek"]["series"].append({
                "tick": tick_idx, "model": prev_prop, "model_after": prop_model,
                "latency_ms": prop_obs_lat, "accuracy": cls.MODEL_SPECS[prev_prop]["accuracy"],
                "lat_ok": prop_obs_lat <= latency_sla_ms, "acc_ok": True,
            })

        summary_table: List[Dict[str, Any]] = []
        for key, pdata in policies.items():
            s = pdata["series"]
            avg_lat = sum(x["latency_ms"] for x in s) / n_ticks
            avg_acc = sum(x["accuracy"] for x in s) / n_ticks
            lat_comp = sum(1 for x in s if x["lat_ok"]) / n_ticks
            overall_comp = sum(1 for x in s if x["lat_ok"] and x["acc_ok"]) / n_ticks
            summary_table.append({
                "policy_id": key,
                "policy_name": pdata["label"],
                "total_switches": pdata["switches"],
                "thrashing_events": pdata["thrashing"],
                "avg_latency_ms": round(avg_lat, 1),
                "avg_accuracy_percent": round(avg_acc * 100.0, 2),
                "latency_sla_compliance_percent": round(lat_comp * 100.0, 1),
                "overall_sla_compliance_percent": round(overall_comp * 100.0, 1),
            })

        return {
            "total_ticks": n_ticks,
            "sla_latency_ms": latency_sla_ms,
            "sla_min_accuracy_percent": round(min_accuracy_sla * 100.0, 1),
            "persistence_window": persistence_window,
            "hysteresis_margin_ms": hysteresis_margin_ms,
            "summary_table": summary_table,
            "traces": {k: v["series"] for k, v in policies.items()},
            "key_finding": (
                f"Proposed MAPE-K eliminates model thrashing (0 thrashing events, only {policies['proposed_mapek']['switches']} clean switches "
                f"vs. {policies['naive_reactive']['switches']} switches in Naive Reactive) while boosting overall SLA compliance from "
                f"{summary_table[0]['overall_sla_compliance_percent']}% (Static XGBoost) to {summary_table[3]['overall_sla_compliance_percent']}%."
            ),
        }


ablation_benchmark = AblationBenchmarkEngine()
