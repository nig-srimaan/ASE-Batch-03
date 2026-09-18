import httpx
import time
import random

API_URL = "http://127.0.0.1:8000/predict"

def generate_features():
    # Generate 20 random float features matching the model input
    return [random.random() for _ in range(20)]

print("Phase 1: Simulating Normal Traffic (1 req/sec)...")
for _ in range(5):
    httpx.post(API_URL, json={"features": generate_features()})
    time.sleep(1)

print("\n🚀 Phase 2: SIMULATING TRAFFIC SPIKE (Flooding the API!)...")
print("This will spike the latency in the API metrics.")
for _ in range(50):
    httpx.post(API_URL, json={"features": generate_features()})
    # Minimal sleep to simulate heavy concurrency and CPU strain
    time.sleep(0.01) 

print("\nPhase 3: Traffic Spike Over. Returning to normal traffic...")
for _ in range(10):
    httpx.post(API_URL, json={"features": generate_features()})
    time.sleep(1)
    
print("Simulation complete.")
