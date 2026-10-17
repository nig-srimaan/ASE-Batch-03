"""
Expanded 10-Test Verification Suite for the Self-Adaptive ML Model-Switching System (v3.0).
"""
import unittest
from backend.app.main import (
    reset_system_state,
    run_viva_scenario,
    get_full_system_status,
    make_prediction,
    get_ablation_benchmark,
    get_traffic_presets,
    prometheus_metrics,
)
from backend.app.schemas import ScenarioRunRequest, PredictionRequest
from backend.app.modules.c10_model_switching import switching_manager


class TestAdaptiveMLSystemV3(unittest.TestCase):

    def setUp(self):
        reset_system_state(clear_history=True)

    def test_01_twelve_components_registered_and_healthy(self):
        status = get_full_system_status()
        self.assertEqual(len(status["components_registry"]), 12)
        self.assertEqual(status["active_model"]["id"], "xgboost")
        self.assertAlmostEqual(status["active_model"]["accuracy"], 0.92, places=2)

    def test_02_temporary_fluctuation_does_not_switch(self):
        res = run_viva_scenario(ScenarioRunRequest(scenario_id="temporary_fluctuation"))
        for step in res["steps_log"]:
            self.assertEqual(step["active_model_after"], "xgboost")
        self.assertGreaterEqual(
            res["system_status"]["stability"]["transient_fluctuations_absorbed"], 2
        )

    def test_03_persistent_violation_switches_xgboost_to_random_forest(self):
        res = run_viva_scenario(ScenarioRunRequest(scenario_id="persistent_violation"))
        models_after = [s["active_model_after"] for s in res["steps_log"]]
        self.assertEqual(models_after[:4], ["xgboost", "xgboost", "xgboost", "xgboost"])
        self.assertEqual(models_after[4], "random_forest")

    def test_04_accuracy_sla_blocks_logistic_regression_degradation(self):
        res = run_viva_scenario(ScenarioRunRequest(scenario_id="accuracy_sla_guard"))
        self.assertEqual(res["system_status"]["active_model"]["id"], "random_forest")

    def test_05_bidirectional_hysteresis_recovery_to_xgboost(self):
        res = run_viva_scenario(ScenarioRunRequest(scenario_id="cooldown_recovery"))
        models_after = [s["active_model_after"] for s in res["steps_log"]]
        self.assertEqual(models_after[:4], ["random_forest"] * 4)
        self.assertEqual(models_after[4], "xgboost")

    def test_06_full_15_step_lifecycle(self):
        res = run_viva_scenario(ScenarioRunRequest(scenario_id="full_lifecycle"))
        self.assertEqual(res["steps_executed"], 15)
        self.assertEqual(res["system_status"]["active_model"]["id"], "xgboost")
        self.assertEqual(res["system_status"]["switching_stats"]["total_switches"], 2)

    def test_07_canary_shadow_verification_and_circuit_breaker(self):
        ok, warmup_ms, report = switching_manager.verify_model_readiness("random_forest")
        self.assertTrue(ok)
        self.assertTrue(report["probability_simplex_valid"])
        self.assertEqual(report["circuit_breaker_state"], "CLOSED_HEALTHY")

    def test_08_traffic_presets_and_feature_explainability(self):
        presets = get_traffic_presets()
        self.assertIn("syn_flood_ddos", presets)
        pred = make_prediction(PredictionRequest(preset_id="syn_flood_ddos", induced_latency_ms=78.0))
        self.assertEqual(pred["preset_used"], "SYN Flood DDoS Attack Burst")
        self.assertEqual(len(pred["top_feature_attributions"]), 5)

    def test_09_ewma_latency_forecaster_and_prometheus_metrics(self):
        make_prediction(PredictionRequest(induced_latency_ms=82.0))
        make_prediction(PredictionRequest(induced_latency_ms=88.0))
        status = get_full_system_status()
        self.assertIn("ewma_latency_ms", status["metrics"])
        self.assertIn("forecast_next_latency_ms", status["metrics"])
        prom = prometheus_metrics()
        self.assertIn("adaptive_ml_ewma_forecast_latency_ms", prom)

    def test_10_ablation_benchmark_proves_zero_thrashing(self):
        study = get_ablation_benchmark()
        by_id = {r["policy_id"]: r for r in study["summary_table"]}
        # Proposed MAPE-K has 0 thrashing events and significantly fewer switches than Naive Reactive
        self.assertEqual(by_id["proposed_mapek"]["thrashing_events"], 0)
        self.assertLess(by_id["proposed_mapek"]["total_switches"], by_id["naive_reactive"]["total_switches"])
        self.assertGreater(
            by_id["proposed_mapek"]["overall_sla_compliance_percent"],
            by_id["static_xgboost"]["overall_sla_compliance_percent"],
        )


if __name__ == "__main__":
    unittest.main()
