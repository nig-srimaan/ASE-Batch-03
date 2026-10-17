# Deep-Dive: 12-Component Self-Adaptive ML Architecture

This document maps every architectural block from the design specification directly to its implementation module:

1. **Data Management Module** (`backend/app/modules/c01_data_management.py`): Generates and preprocesses a 20-feature Cybersecurity Network Intrusion dataset, performing Z-score standardization and stratified 75/25 splitting.
2. **Multiple ML Models** (`backend/app/modules/c02_ml_models.py`): Trains and manages XGBoost (`92%` accuracy, `80ms` latency, `65%` CPU), Random Forest (`88%` accuracy, `60ms` latency, `50%` CPU), and Logistic Regression (`82%` accuracy, `20ms` latency, `30%` CPU).
3. **Model Evaluation Module** (`backend/app/modules/c03_model_evaluation.py`): Evaluates candidate models prior to deployment and writes baseline profiles to `saved_models/model_profiles.json` and the Knowledge Base.
4. **Prediction Service** (`backend/app/modules/c04_prediction_service.py`): Serves real-time predictions via `POST /api/predict` using whichever model is currently active.
5. **Runtime Monitoring Module** (`backend/app/modules/c05_runtime_monitoring.py`): Collects sliding-window latency (`p50`, `p95`, `p99`), CPU, memory, and request rate telemetry, and exports Prometheus metrics at `GET /metrics`.
6. **SLA Manager** (`backend/app/modules/c06_sla_manager.py`): Enforces `Accuracy >= 85%`, `Latency <= 100 ms`, and `CPU <= 70%`, and projects candidate performance under current load.
7. **MAPE-K Adaptation Engine** (`backend/app/modules/c07_mapek_engine.py`): Orchestrates the explicit `Monitor -> Analyze -> Plan -> Execute -> Knowledge` feedback loop.
8. **Adaptive Model Selector** (`backend/app/modules/c08_model_selector.py`): Filters out candidate models violating hard SLA constraints and selects the highest-utility feasible model.
9. **Stability & Hysteresis Manager** (`backend/app/modules/c09_stability_hysteresis.py`): Distinguishes transient fluctuations from persistent violations (`5` consecutive breaches) and prevents thrashing during recovery using cooldown and hysteresis margins.
10. **Model Switching / Deployment Manager** (`backend/app/modules/c10_model_switching.py`): Performs warm-up readiness verification and zero-downtime atomic pointer switching.
11. **Knowledge Base** (`backend/app/modules/c11_knowledge_base.py`): Relational persistence store (`SQLite` & `PostgreSQL`) storing model profiles, telemetry snapshots, and adaptation logs.
12. **Dashboard / User Interface** (`frontend/`): Real-time React & Web UI visualizing live telemetry, SLA boundaries, MAPE-K traces, and 1-click Viva scenarios.
