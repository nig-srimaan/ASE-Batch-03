# Self-Adaptive ML Model-Switching System (12-Component MAPE-K Architecture)

An end-to-end **Adaptive Software Engineering (ASE)** system that performs **bidirectional runtime ML model switching** under dynamic workload and Service Level Agreement (SLA) constraints using the classical **MAPE-K (Monitor–Analyze–Plan–Execute–Knowledge)** feedback loop.

> **"The system performs bidirectional adaptation: it switches to a resource-efficient model during sustained resource pressure and switches back to a higher-performing model when normal conditions are restored and remain stable."**

---

## Core Adaptive Capabilities

1. **Multi-Model Accuracy–Latency–Resource Trade-Off Pool**:
   | Model | Tier | Baseline Accuracy | Baseline Latency | Baseline CPU | Role in Adaptation |
   | :--- | :--- | :---: | :---: | :---: | :--- |
   | **XGBoost** | `HEAVY` | **92.0%** | `80 ms` | `65%` | Default optimal accuracy model under normal workload |
   | **Random Forest** | `BALANCED` | **88.0%** | `60 ms` | `50%` | Resilient fallback satisfying both Accuracy (`>=85%`) & Latency (`<=100ms`) SLAs |
   | **Logistic Regression** | `LIGHT` | **82.0%** | `20 ms` | `30%` | Ultra-low latency model (rejected when strict `85%` Accuracy SLA floor is enforced) |

2. **Persistent SLA Violation Filtering (No False Switches on Temporary Spikes)**:
   - **Case 1 (Temporary Fluctuation)**: `90 -> 105 -> 98 -> 103 -> 95 ms`
     - Latency briefly crosses `100 ms` on single ticks but recovers immediately.
     - The **Stability & Hysteresis Manager** absorbs these transient spikes and **does not switch models**.
   - **Case 2 (Persistent Violation)**: `105 -> 112 -> 120 -> 130 -> 125 ms`
     - Latency stays continuously above the `100 ms` SLA for the configured persistence window (`5` consecutive breaches).
     - Triggers the **MAPE-K Adaptation Engine** to evaluate alternatives and switch `XGBoost -> Random Forest`.

3. **Multi-Criteria SLA Constraint Verification**:
   - The **Adaptive Model Selector** never switches blindly based on latency alone.
   - When `XGBoost` violates latency (`130 ms > 100 ms`), the selector checks all alternatives against hard SLA constraints (`Accuracy >= 85%`, `Latency <= 100 ms`, `CPU <= 70%`).
   - `Logistic Regression` (`20 ms` latency, `82%` accuracy) is **rejected** because `82% < 85%` violates the Accuracy SLA.
   - `Random Forest` (`88%` accuracy, `~65 ms` projected latency) satisfies all SLAs and is selected.

4. **Bidirectional Adaptation with Cooldown & Anti-Thrashing Hysteresis**:
   - When high workload subsides, the system **does not immediately switch back** (which would cause `XGBoost <-> Random Forest` oscillation/thrashing).
   - Instead, it waits for:
     1. Post-switch **Cooldown Timer** expiration,
     2. Sustained **Recovery Stability Window** (`5` consecutive stable samples), and
     3. **Hysteresis Safety Margin** verification (`Projected XGBoost Latency <= SLA - Hysteresis Margin`, i.e., `<= 88 ms`).
   - Once verified, it cleanly upgrades back `Random Forest -> XGBoost`.

---

## Complete 12-Component Architecture Mapping

| # | Component Name | Source File | Primary Responsibility |
| :-: | :--- | :--- | :--- |
| **1** | **Data Management Module** | `backend/app/modules/c01_data_management.py` | Dataset generation/loading, NaN imputation, Z-score standardization, stratified train/test split |
| **2** | **Multiple ML Models** | `backend/app/modules/c02_ml_models.py` | Trains & manages XGBoost (`HEAVY`), Random Forest (`BALANCED`), and Logistic Regression (`LIGHT`) |
| **3** | **Model Evaluation Module** | `backend/app/modules/c03_model_evaluation.py` | Pre-deployment benchmarking of Accuracy, F1, Precision, Recall, Latency, CPU, and Memory |
| **4** | **Prediction Service** | `backend/app/modules/c04_prediction_service.py` | Real-time inference routing requests to the currently active model |
| **5** | **Runtime Monitoring Module** | `backend/app/modules/c05_runtime_monitoring.py` | Sliding-window telemetry (`p50`, `p95`, `p99` latency, CPU, memory, req/min) & Prometheus `/metrics` |
| **6** | **SLA Manager** | `backend/app/modules/c06_sla_manager.py` | Enforces `Accuracy >= 85%`, `Latency <= 100 ms`, `CPU <= 70%` & projects candidate compliance |
| **7** | **MAPE-K Adaptation Engine** | `backend/app/modules/c07_mapek_engine.py` | Closed-loop orchestrator (`Monitor -> Analyze -> Plan -> Execute -> Knowledge`) |
| **8** | **Adaptive Model Selector** | `backend/app/modules/c08_model_selector.py` | Filters out SLA-violating models and ranks feasible candidates by accuracy & utility |
| **9** | **Stability & Hysteresis Manager** | `backend/app/modules/c09_stability_hysteresis.py` | Absorbs temporary fluctuations and enforces cooldown + hysteresis margin on switch-back |
| **10** | **Model Switching Manager** | `backend/app/modules/c10_model_switching.py` | Verifies candidate readiness via warm-up inference and performs atomic zero-downtime hot-swap |
| **11** | **Knowledge Base** | `backend/app/modules/c11_knowledge_base.py` | Persists model profiles, telemetry time-series, and adaptation audit logs (PostgreSQL / SQLite) |
| **12** | **Dashboard / User Interface** | `frontend/src/App.jsx` & `frontend/dist/index.html` | Interactive real-time dashboard with live SLA charts, MAPE-K inspector, and 1-click Viva scenarios |

---

## Running 100% on GitHub (Zero Local Setup via GitHub Codespaces)

1. Create a new repository on GitHub and upload/commit the contents of this archive.
2. Click the green **`<> Code`** button on your GitHub repository -> select the **Codespaces** tab -> click **Create codespace on main**.
3. The included `.devcontainer/devcontainer.json` automatically installs dependencies, trains the models, starts the FastAPI + Dashboard server on port `8000`, and opens the live browser preview.

### Terminal Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train & evaluate the 3 ML models
python3 -m backend.app.cli train

# 3. Run the 15-step bidirectional adaptation CLI demonstration
python3 -m backend.app.cli demo

# 4. Run the automated verification test suite (6 unit/integration tests)
python3 -m unittest discover -s tests -v

# 5. Start the FastAPI server + Interactive Web Dashboard on http://localhost:8000
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```
