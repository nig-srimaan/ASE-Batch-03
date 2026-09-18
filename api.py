from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import time
import psutil
import numpy as np

app = FastAPI(title="Adaptive ML Inference API")

# Load models into memory
models = {
    "logistic": joblib.load("model_logistic.pkl"),
    "random_forest": joblib.load("model_rf.pkl"),
    "xgboost": joblib.load("model_xgboost.pkl")
}

# State Variables - Start with the heaviest/best model
current_model_name = "xgboost"  
metrics_history = []

class PredictRequest(BaseModel):
    features: list[float]

class SwitchRequest(BaseModel):
    model_name: str

@app.post("/predict")
def predict(req: PredictRequest):
    global current_model_name, metrics_history
    
    start_time = time.time()
    
    # Run prediction
    model = models[current_model_name]
    input_data = np.array(req.features).reshape(1, -1)
    
    # Artificial delay to simulate processing weight across different models
    if current_model_name == "xgboost":
        time.sleep(0.08) # 80ms base
    elif current_model_name == "random_forest":
        time.sleep(0.04) # 40ms base
    elif current_model_name == "logistic":
        time.sleep(0.01) # 10ms base
        
    prediction = int(model.predict(input_data)[0])
    latency_ms = (time.time() - start_time) * 1000
    
    # Track runtime metrics for the SLA manager (keep last 50 requests)
    cpu_usage = psutil.cpu_percent()
    metrics_history.append({"latency_ms": latency_ms, "cpu": cpu_usage})
    if len(metrics_history) > 50:
        metrics_history.pop(0)
        
    return {
        "model_used": current_model_name,
        "prediction": prediction,
        "latency_ms": round(latency_ms, 2)
    }

@app.get("/metrics")
def get_metrics():
    if not metrics_history:
        return {"status": "No traffic yet", "active_model": current_model_name}
    
    avg_latency = sum(m["latency_ms"] for m in metrics_history) / len(metrics_history)
    avg_cpu = sum(m["cpu"] for m in metrics_history) / len(metrics_history)
    
    return {
        "active_model": current_model_name,
        "avg_latency_ms": round(avg_latency, 2),
        "avg_cpu_percent": round(avg_cpu, 2)
    }

@app.post("/switch_model")
def switch_model(req: SwitchRequest):
    global current_model_name
    if req.model_name not in models:
        raise HTTPException(status_code=400, detail="Invalid model name")
    
    current_model_name = req.model_name
    return {"message": f"Successfully switched to {current_model_name}"}
