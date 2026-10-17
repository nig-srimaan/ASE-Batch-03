# Self-Adaptive Machine Learning Inference Under Dynamic Workloads: A 12-Component Bidirectional MAPE-K Architecture with Hysteresis-Governed Stability

## Abstract
Modern machine learning (ML) inference services face a fundamental operational tension between predictive accuracy and real-time latency under fluctuating network workloads. High-capacity ensemble models such as XGBoost deliver superior accuracy (`92.0%`) during normal operating conditions (`80 ms` latency), but suffer severe Service Level Agreement (SLA) violations (`>100 ms`) during sustained traffic surges. Conversely, lightweight linear classifiers such as Logistic Regression satisfy strict latency budgets (`20 ms`) at the expense of violating accuracy guarantees (`82.0% < 85.0%` SLA). Naive threshold-based model switching introduces severe **model thrashing (oscillation)** whenever workload fluctuates near the SLA boundary. In this project, we design and implement a **12-Component Self-Adaptive ML Model-Switching System** governed by a closed-loop **MAPE-K (Monitor–Analyze–Plan–Execute–Knowledge)** architecture. Our system introduces (1) a **Persistent Violation Filter** that absorbs transient fluctuations, (2) a **Multi-Criteria Constraint-Aware Model Selector** that protects hard accuracy floors while optimizing latency and CPU headroom, (3) **Bidirectional Adaptation with Cooldown and Hysteresis Margins** to eliminate model oscillation during workload recovery, and (4) **Pre-Switch Canary Shadow Verification** for zero-downtime model transitions.

---

## 1. Mathematical Formulation of Adaptation Rules

### 1.1 Service Level Agreement (SLA) Vector
Let the operational SLA be defined by the tuple:
$$\text{SLA} = \langle A_{\min}, L_{\max}, C_{\max} \rangle = \langle 0.85, 100\text{ ms}, 70\% \rangle$$

A model $m \in \mathcal{M} = \{\text{XGBoost}, \text{RandomForest}, \text{LogisticRegression}\}$ at time step $t$ under workload $W_t$ is **SLA-compliant** if and only if:
$$\text{Compliant}(m, t) \iff A(m) \ge A_{\min} \;\land\; L(m, W_t) \le L_{\max} \;\land\; C(m, W_t) \le C_{\max}$$

### 1.2 Persistent Violation Detection (Downward Adaptation)
To prevent reacting to transient noise (`90 -> 105 -> 98 -> 103 -> 95 ms`), the consecutive violation counter $V_t$ evolves as:
$$V_t = \begin{cases} V_{t-1} + 1 & \text{if } \neg \text{Compliant}(m_{\text{active}}, t) \\ 0 & \text{if } \text{Compliant}(m_{\text{active}}, t) \end{cases}$$

A downward adaptation is triggered only when $V_t \ge K_{\text{persist}}$ (where $K_{\text{persist}} = 5$ consecutive ticks).

### 1.3 Multi-Criteria Feasible Model Selection
Given the feasible candidate set $\mathcal{F}_t = \{ m \in \mathcal{M} \mid A(m) \ge A_{\min} \land \hat{L}(m, W_t) \le L_{\max} \land \hat{C}(m, W_t) \le C_{\max} \}$, the **Adaptive Model Selector** ranks candidates by composite utility:
$$U(m, W_t) = w_A \cdot A(m) + w_L \cdot \frac{L_{\max} - \hat{L}(m, W_t)}{L_{\max}} + w_C \cdot \frac{C_{\max} - \hat{C}(m, W_t)}{C_{\max}}$$
with $w_A = 0.70, w_L = 0.20, w_C = 0.10$.

### 1.4 Bidirectional Recovery with Anti-Thrashing Hysteresis (Upward Adaptation)
When operating on a degraded model (e.g., `RandomForest`) after a high-load event, switching back to the superior model $m^* = \text{XGBoost}$ requires all three stability conditions:
1. **Cooldown Expiration**: $t - t_{\text{last\_switch}} \ge T_{\text{cooldown}}$
2. **Sustained Stability Window**: $S_t \ge K_{\text{stable}}$ ($5$ consecutive compliant ticks)
3. **Hysteresis Safety Margin**:
$$\hat{L}(m^*, W_t) \le L_{\max} - \Delta_{\text{hyst}} \quad (100\text{ ms} - 12\text{ ms} = 88\text{ ms})$$

---

## 2. Experimental Ablation Results (30-Tick Multi-Phase Workload)

| Policy | Total Switches | Thrashing Events | Mean Latency (ms) | Mean Accuracy (%) | Latency SLA Compliance | Overall SLA Compliance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Static XGBoost (No Adaptation)** | `0` | `0` | `101.6 ms` | `92.00%` | `40.0%` | `40.0%` |
| **Static Logistic Regression** | `0` | `0` | `21.8 ms` | `82.00%` | `100.0%` | `0.0%` *(Violates 85% Acc)* |
| **Naive Reactive Switcher** | `10` | `6` | `85.1 ms` | `89.60%` | `60.0%` | `60.0%` |
| **Proposed MAPE-K (Ours)** | **`2`** | **`0`** | **`76.6 ms`** | **`89.73%`** | **`80.0%`** | **`80.0%`** |
