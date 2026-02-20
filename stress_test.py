import time
import requests
import random
import threading

API_URL = "http://localhost:8000"

def send_traffic():
    endpoints = ["/predict", "/health", "/"]
    while True:
        try:
            ep = random.choice(endpoints)
            if ep == "/predict":
                payload = {"article": "BTC-USDT", "days_ahead": 1}
                requests.post(f"{API_URL}{ep}", json=payload)
            else:
                requests.get(f"{API_URL}{ep}")
                
            time.sleep(random.uniform(0.1, 0.5))
            print(f"✅ Successfully processed request: {ep}") # Fixed print statement!
            
        except Exception as e:
            print(f"❌ Server offline: {e}")
            time.sleep(1)

if __name__ == "__main__":
    print(f"🚀 Launching Quant Traffic Simulator against {API_URL}...")
    for _ in range(5):
        t = threading.Thread(target=send_traffic)
        t.daemon = True
        t.start()
    while True:
        time.sleep(1)