"""
Command-Line Interface for training, evaluating, demonstrating, and benchmarking the
12-Component Self-Adaptive ML Model-Switching System (v3.0).
Usage:
    python -m backend.app.cli train
    python -m backend.app.cli demo
    python -m backend.app.cli benchmark
"""
import sys
from backend.app.modules.c03_model_evaluation import model_evaluator
from backend.app.modules.c10_model_switching import switching_manager
from backend.app.modules.c13_ablation_benchmark import ablation_benchmark
from backend.app.main import run_viva_scenario, reset_system_state
from backend.app.schemas import ScenarioRunRequest


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if cmd == "train":
        print("=== Training & Evaluating All 3 Candidate ML Models ===")
        profiles = model_evaluator.run_full_evaluation(force_retrain=True)
        switching_manager.ensure_initialized()
        for mid, p in profiles.items():
            print(
                f"  [{p['tier']:8s}] {p['name']:20s} | "
                f"Accuracy: {p['accuracy']*100:5.1f}% (Empirical: {p['empirical_test_accuracy']*100:5.1f}%) | "
                f"Baseline Latency: {p['baseline_latency_ms']:5.1f}ms | CPU: {p['baseline_cpu_percent']:4.1f}%"
            )
        print("Saved model bundle and Knowledge Base profiles successfully.")
    elif cmd == "demo":
        print("=== Running End-to-End MAPE-K Bidirectional Adaptation Lifecycle ===")
        reset_system_state(clear_history=True)
        res = run_viva_scenario(ScenarioRunRequest(scenario_id="full_lifecycle"))
        for i, step in enumerate(res["steps_log"], 1):
            print(
                f"Step {i:02d} | {step['step_note']:48s} | "
                f"Model: {step['active_model_before']} -> {step['active_model_after']} | "
                f"State: {step['stability_state']}"
            )
    elif cmd == "benchmark":
        print("=== Running 30-Tick Comparative Policy Ablation Study ===")
        study = ablation_benchmark.run_comparative_study()
        for row in study["summary_table"]:
            print(
                f"  {row['policy_name']:52s} | Switches: {row['total_switches']:2d} (Thrash: {row['thrashing_events']:2d}) | "
                f"Avg Lat: {row['avg_latency_ms']:5.1f}ms | Avg Acc: {row['avg_accuracy_percent']:5.2f}% | "
                f"SLA Compliance: {row['overall_sla_compliance_percent']:5.1f}%"
            )
        print("Key Finding:", study["key_finding"])
    else:
        print(f"Unknown command: {cmd}. Use 'train', 'demo', or 'benchmark'.")


if __name__ == "__main__":
    main()
