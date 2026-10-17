"""
FastAPI Server & Orchestrator for the 12-Component Self-Adaptive ML Model-Switching System (v3.0).
Provides:
- `/api/predict`            (Component 4: Prediction Service + Feature Attribution + Traffic Presets)
- `/api/presets`            (Curated Cybersecurity Network Flow Presets)
- `/api/status`             (Complete 12-component state, active model, SLA status, EWMA telemetry, MAPE-K trace)
- `/api/benchmark/compare`  (Ablation study comparing Static, Naive Reactive Switcher, and Proposed MAPE-K)
- `/api/mapek/step`         (Trigger single MAPE-K loop cycle)
- `/api/workload`           (Adjust live traffic workload req/min)
- `/api/sla`                (Update SLA thresholds dynamically)
- `/api/stability`          (Update persistence window, cooldown, and hysteresis margin)
- `/api/scenarios/run`      (Execute deterministic Viva demonstration scenarios)
- `/api/reset`              (Reset system back to baseline XGBoost @ 60 req/min)
- `/metrics`                (Prometheus scrape endpoint)
- `/`                       (Serves the 5-Tab Interactive Command Center Dashboard UI)
"""
import os
from contextlib import asynccontextmanager
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.config import config
from backend.app.schemas import (
    PredictionRequest,
    PredictionResponse,
    SLAConfigUpdate,
    StabilityConfigUpdate,
    WorkloadUpdate,
    ScenarioRunRequest,
)
from backend.app.modules.c01_data_management import data_manager
from backend.app.modules.c02_ml_models import ml_models_module
from backend.app.modules.c03_model_evaluation import model_evaluator
from backend.app.modules.c04_prediction_service import prediction_service, TRAFFIC_PRESETS
from backend.app.modules.c05_runtime_monitoring import runtime_monitor
from backend.app.modules.c06_sla_manager import sla_manager
from backend.app.modules.c07_mapek_engine import mapek_engine
from backend.app.modules.c08_model_selector import model_selector
from backend.app.modules.c09_stability_hysteresis import stability_manager
from backend.app.modules.c10_model_switching import switching_manager
from backend.app.modules.c11_knowledge_base import knowledge_base
from backend.app.modules.c13_ablation_benchmark import ablation_benchmark


@asynccontextmanager
async def lifespan(app: FastAPI):
    switching_manager.ensure_initialized()
    if not mapek_engine.last_cycle_trace:
        prediction_service.predict(induced_latency_ms=80.0)
        mapek_engine.execute_cycle()
    yield


app = FastAPI(
    title=config.project_name,
    version="3.0.0",
    description="12-Component Self-Adaptive ML Inference & Model-Switching System using the MAPE-K Feedback Loop",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/predict", response_model=PredictionResponse, tags=["Prediction Service & Explainability"])
def make_prediction(req: PredictionRequest):
    """Serves a prediction using the currently active ML model, computes feature attribution, and runs MAPE-K."""
    res = prediction_service.predict(
        features=req.features,
        workload_multiplier=req.workload_multiplier,
        induced_latency_ms=req.induced_latency_ms,
        preset_id=req.preset_id,
    )
    mapek_engine.execute_cycle()
    return res


@app.get("/api/presets", tags=["Prediction Service & Explainability"])
def get_traffic_presets() -> Dict[str, Any]:
    """Returns curated network flow presets for interactive testing."""
    return TRAFFIC_PRESETS


@app.get("/api/benchmark/compare", tags=["Ablation Study & Evaluation"])
def get_ablation_benchmark() -> Dict[str, Any]:
    """
    Runs a 30-tick comparative benchmark across:
    1. Static XGBoost
    2. Static Logistic Regression
    3. Naive Reactive Switcher (No Persistence / No Hysteresis -> Thrashing)
    4. Proposed MAPE-K (Persistent SLA Window + Bidirectional Hysteresis)
    """
    return ablation_benchmark.run_comparative_study(
        latency_sla_ms=sla_manager.max_latency_sla_ms,
        min_accuracy_sla=sla_manager.min_accuracy_sla,
        persistence_window=stability_manager.persistent_violation_window,
        hysteresis_margin_ms=stability_manager.hysteresis_latency_margin_ms,
    )


@app.post("/api/mapek/step", tags=["MAPE-K Engine"])
def trigger_mapek_step(ignore_cooldown: bool = False):
    """Generates a live prediction under current workload and executes one MAPE-K cycle."""
    pred = prediction_service.predict()
    trace = mapek_engine.execute_cycle(ignore_cooldown=ignore_cooldown)
    return {
        "prediction": pred,
        "mapek_trace": trace,
    }


@app.get("/api/status", tags=["System Telemetry & 12 Components"])
def get_full_system_status() -> Dict[str, Any]:
    """Returns complete state across all 12 architectural components for the dashboard."""
    switching_manager.ensure_initialized()
    active_id, _, active_profile = switching_manager.get_active_model_bundle()
    all_profiles = model_evaluator.get_profiles()
    metrics = runtime_monitor.get_metrics_snapshot()
    sla_eval = sla_manager.evaluate_current_compliance(active_profile, metrics)
    stability_state = stability_manager.get_status()
    kb_summary = knowledge_base.get_knowledge_summary()
    adaptations = knowledge_base.get_adaptation_history(limit=25)
    telemetry_series = knowledge_base.get_telemetry_history(limit=60)

    _, candidate_evaluations, rationale = model_selector.select_best_model(
        active_model_id=active_id,
        all_profiles=all_profiles,
        current_workload_req_per_min=metrics["workload_req_per_min"],
        observed_latency_ms=metrics["latest_latency_ms"],
    )

    return {
        "active_model": active_profile,
        "metrics": metrics,
        "sla": sla_eval,
        "stability": stability_state,
        "candidate_evaluations": candidate_evaluations,
        "selector_rationale": rationale,
        "last_mapek_trace": mapek_engine.last_cycle_trace,
        "recent_adaptations": adaptations,
        "telemetry_history": telemetry_series,
        "knowledge_base": kb_summary,
        "dataset": data_manager.get_status(),
        "traffic_presets": TRAFFIC_PRESETS,
        "ablation_benchmark": ablation_benchmark.run_comparative_study(
            latency_sla_ms=sla_manager.max_latency_sla_ms,
            min_accuracy_sla=sla_manager.min_accuracy_sla,
            persistence_window=stability_manager.persistent_violation_window,
            hysteresis_margin_ms=stability_manager.hysteresis_latency_margin_ms,
        ),
        "switching_stats": {
            "total_switches": switching_manager.switch_count,
            "active_model_id": active_id,
            "last_switch_timestamp": switching_manager.last_switch_timestamp,
            "canary_checks_passed": switching_manager.canary_checks_passed,
            "circuit_breaker_trips": switching_manager.circuit_breaker_trips,
            "last_canary_report": switching_manager.last_canary_report,
        },
        "components_registry": [
            {"id": 1, "name": "Data Management Module", "file": "backend/app/modules/c01_data_management.py", "status": "HEALTHY", "role": "20-feature Cyber Telemetry Standardization & 75/25 Split"},
            {"id": 2, "name": "Multiple ML Models", "file": "backend/app/modules/c02_ml_models.py", "status": "3 MODELS READY", "role": "XGBoost (92%), Random Forest (88%), Logistic Regression (82%)"},
            {"id": 3, "name": "Model Evaluation Module", "file": "backend/app/modules/c03_model_evaluation.py", "status": "BENCHMARKED", "role": "Pre-deployment Accuracy, F1, Latency & CPU Profiling"},
            {"id": 4, "name": "Prediction Service", "file": "backend/app/modules/c04_prediction_service.py", "status": "ACTIVE + XAI", "role": "Live Inference + Top-5 Local Feature Attribution"},
            {"id": 5, "name": "Runtime Monitoring Module", "file": "backend/app/modules/c05_runtime_monitoring.py", "status": "EWMA + PROMETHEUS", "role": "Sliding-Window p50/p95/p99, EWMA Forecast & /metrics"},
            {"id": 6, "name": "SLA Manager", "file": "backend/app/modules/c06_sla_manager.py", "status": "COMPLIANT" if sla_eval["sla_satisfied"] else "VIOLATING", "role": "Acc >= 85%, Latency <= 100ms, CPU <= 70% Enforcement"},
            {"id": 7, "name": "MAPE-K Adaptation Engine", "file": "backend/app/modules/c07_mapek_engine.py", "status": f"CYCLE #{mapek_engine.cycle_count}", "role": "Closed-Loop Monitor -> Analyze -> Plan -> Execute -> Knowledge"},
            {"id": 8, "name": "Adaptive Model Selector", "file": "backend/app/modules/c08_model_selector.py", "status": "MULTI-CRITERIA", "role": "Hard SLA Constraint Filter + Composite Utility Ranking"},
            {"id": 9, "name": "Stability & Hysteresis Manager", "file": "backend/app/modules/c09_stability_hysteresis.py", "status": stability_state["state"], "role": "Persistence Window (5 ticks) + Cooldown + Hysteresis Margin"},
            {"id": 10, "name": "Model Switching Manager", "file": "backend/app/modules/c10_model_switching.py", "status": f"SERVING {active_profile['name'].upper()}", "role": "5-Sample Canary Shadow Verification + Atomic Hot-Swap"},
            {"id": 11, "name": "Knowledge Base (PostgreSQL/SQLite)", "file": "backend/app/modules/c11_knowledge_base.py", "status": f"{kb_summary['total_adaptations_logged']} ADAPTATIONS", "role": "Relational Audit Store of Profiles, Traces & Telemetry"},
            {"id": 12, "name": "Dashboard / User Interface", "file": "frontend/src/App.jsx", "status": "5-TAB COMMAND UI", "role": "Real-Time Charts, Viva Scenarios, Ablation & Packet Lab"},
        ],
    }


@app.post("/api/workload", tags=["Controls & Simulation"])
def update_workload(req: WorkloadUpdate):
    runtime_monitor.set_workload(req.req_per_min, req.noise_std_ms)
    pred = prediction_service.predict()
    trace = mapek_engine.execute_cycle()
    return {"workload_req_per_min": req.req_per_min, "prediction": pred, "mapek_trace": trace}


@app.post("/api/sla", tags=["Controls & Simulation"])
def update_sla_thresholds(req: SLAConfigUpdate):
    updated = sla_manager.update_sla(
        min_accuracy_sla=req.min_accuracy_sla,
        max_latency_sla_ms=req.max_latency_sla_ms,
        max_cpu_sla_percent=req.max_cpu_sla_percent,
        allow_emergency_degrade=req.allow_emergency_degrade_below_accuracy_sla,
    )
    trace = mapek_engine.execute_cycle(ignore_cooldown=True)
    return {"sla": updated, "mapek_trace": trace}


@app.post("/api/stability", tags=["Controls & Simulation"])
def update_stability_settings(req: StabilityConfigUpdate):
    updated = stability_manager.update_config(
        persistent_violation_window=req.persistent_violation_window,
        recovery_stability_window=req.recovery_stability_window,
        cooldown_seconds=req.cooldown_seconds,
        hysteresis_latency_margin_ms=req.hysteresis_latency_margin_ms,
    )
    return {"stability": updated}


@app.post("/api/reset", tags=["Controls & Simulation"])
def reset_system_state(clear_history: bool = False):
    switching_manager.force_set_model("xgboost")
    sla_manager.update_sla(min_accuracy_sla=0.85, max_latency_sla_ms=100.0, max_cpu_sla_percent=70.0, allow_emergency_degrade=False)
    stability_manager.reset_counters()
    runtime_monitor.set_workload(60.0, noise_std_ms=1.0)
    runtime_monitor.reset_window(initial_latency_ms=80.0, initial_cpu_percent=65.0)
    if clear_history:
        knowledge_base.clear_runtime_logs()
        switching_manager.switch_count = 0
    prediction_service.predict(induced_latency_ms=80.0)
    mapek_engine.execute_cycle(ignore_cooldown=True)
    return get_full_system_status()


@app.post("/api/scenarios/run", tags=["Viva Demonstration Scenarios"])
def run_viva_scenario(req: ScenarioRunRequest) -> Dict[str, Any]:
    steps_log: List[Dict[str, Any]] = []

    def _step(lat_ms: float, workload: float, note: str, ignore_cd: bool = True):
        runtime_monitor.set_workload(workload, noise_std_ms=0.0)
        prediction_service.predict(induced_latency_ms=lat_ms)
        trace = mapek_engine.execute_cycle(ignore_cooldown=ignore_cd)
        steps_log.append({
            "step_note": note,
            "injected_latency_ms": lat_ms,
            "workload_req_per_min": workload,
            "active_model_before": trace["monitor"]["active_model_id"],
            "active_model_after": switching_manager.active_model_id,
            "stability_state": trace["analyze"]["stability_state"],
            "consecutive_violations": trace["analyze"]["consecutive_violations"],
            "consecutive_stable": trace["analyze"]["consecutive_stable_samples"],
            "action": trace["execute"]["action"],
            "explanation": trace["analyze"]["stability_explanation"],
        })

    if req.scenario_id == "temporary_fluctuation":
        switching_manager.force_set_model("xgboost")
        stability_manager.reset_counters()
        for idx, lat in enumerate([90.0, 105.0, 98.0, 103.0, 95.0], start=1):
            _step(lat, 65.0, f"Tick {idx}: Latency = {lat:.0f} ms (SLA = 100 ms)")

    elif req.scenario_id == "persistent_violation":
        switching_manager.force_set_model("xgboost")
        stability_manager.reset_counters()
        for idx, lat in enumerate([105.0, 112.0, 120.0, 130.0, 125.0], start=1):
            _step(lat, 135.0, f"Tick {idx}: Sustained High Load Latency = {lat:.0f} ms (> 100 ms SLA)")

    elif req.scenario_id == "accuracy_sla_guard":
        switching_manager.force_set_model("random_forest")
        stability_manager.reset_counters()
        sla_manager.update_sla(min_accuracy_sla=0.85, max_latency_sla_ms=100.0, allow_emergency_degrade=False)
        for idx, lat in enumerate([108.0, 114.0, 118.0, 122.0, 125.0], start=1):
            _step(lat, 260.0, f"Tick {idx}: Extreme Load on RF ({lat:.0f}ms) - Checking if Logistic Regression (82% Acc) is allowed")

    elif req.scenario_id == "cooldown_recovery":
        switching_manager.force_set_model("random_forest")
        stability_manager.reset_counters()
        for idx, lat in enumerate([65.0, 62.0, 60.0, 58.0, 57.0], start=1):
            _step(lat, 55.0, f"Recovery Tick {idx}: Load cooled down, RF Latency = {lat:.0f} ms (Stabilizing before upgrade)")

    elif req.scenario_id == "full_lifecycle":
        switching_manager.force_set_model("xgboost")
        stability_manager.reset_counters()
        for lat in [90.0, 105.0, 98.0, 103.0, 95.0]:
            _step(lat, 65.0, f"Phase 1 (Temporary Fluctuation): {lat:.0f} ms")
        for lat in [105.0, 112.0, 120.0, 130.0, 125.0]:
            _step(lat, 135.0, f"Phase 2 (Persistent Violation): {lat:.0f} ms")
        for lat in [64.0, 61.0, 59.0, 58.0, 57.0]:
            _step(lat, 55.0, f"Phase 3 (Cooldown & Hysteresis Recovery): {lat:.0f} ms")
    else:
        raise HTTPException(status_code=400, detail=f"Unknown scenario_id: {req.scenario_id}")

    return {
        "scenario_id": req.scenario_id,
        "steps_executed": len(steps_log),
        "steps_log": steps_log,
        "system_status": get_full_system_status(),
    }


@app.get("/metrics", response_class=PlainTextResponse, tags=["Monitoring (Prometheus)"])
def prometheus_metrics():
    active_id = switching_manager.active_model_id
    metrics = runtime_monitor.get_metrics_snapshot()
    violated = metrics["latest_latency_ms"] > sla_manager.max_latency_sla_ms
    return runtime_monitor.export_prometheus_metrics(
        active_model_id=active_id,
        sla_violated=violated,
        switch_count=switching_manager.switch_count,
    )


FRONTEND_DIST = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../frontend/dist"))
if os.path.isdir(FRONTEND_DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIST, "assets"), check_dir=False), name="assets")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def serve_dashboard():
        index_file = os.path.join(FRONTEND_DIST, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return HTMLResponse("<h1>Adaptive ML Model-Switching API is running. Visit /docs</h1>")
