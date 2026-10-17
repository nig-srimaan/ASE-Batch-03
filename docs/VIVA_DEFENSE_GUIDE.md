# Academic Viva & Project Defense Guide

### Q1: "Where is the Adaptive Software Engineering (ASE) part in this project?"
**Answer:**
Our system implements a closed-loop **MAPE-K Self-Adaptive Architecture** across 7 dedicated ASE modules:
- **Runtime Monitoring (#5)**, **SLA Manager (#6)**, **MAPE-K Adaptation Engine (#7)**, **Adaptive Model Selector (#8)**, **Stability & Hysteresis Manager (#9)**, **Model Switching Manager (#10)**, and **Knowledge Base (#11)**.

### Q2: "What happens on a temporary latency spike (`90 -> 105 -> 98 -> 103 -> 95 ms`)?"
**Answer:**
The **Stability & Hysteresis Manager (`c09_stability_hysteresis.py`)** requires a persistent violation (`5` consecutive breaches) before triggering a model switch. Single spikes reset the breach counter as soon as latency returns below `100 ms`, logging `TRANSIENT_FLUCTUATION_ABSORBED`.

### Q3: "When `XGBoost` persistently violates the 100 ms SLA (`105 -> 112 -> 120 -> 130 -> 125 ms`), why switch to `Random Forest` instead of `Logistic Regression` (20 ms)?"
**Answer:**
The **Adaptive Model Selector (`c08_model_selector.py`)** checks all SLA constraints simultaneously. `Logistic Regression` has `82%` accuracy, which violates our `Accuracy >= 85%` SLA floor. `Random Forest` satisfies both `Accuracy >= 85%` (`88%`) and `Latency <= 100 ms` (`~65 ms`).

### Q4: "How do you prevent oscillation (thrashing) when workload cools down?"
**Answer:**
Before upgrading back from `Random Forest` to `XGBoost`, the **Stability & Hysteresis Manager** enforces:
1. Cooldown timer expiration,
2. Recovery stability window (`5` consecutive stable ticks), and
3. Hysteresis margin check (`Projected XGBoost Latency <= SLA - Hysteresis Margin = 88 ms`).
