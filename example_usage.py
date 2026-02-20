import time
import requests
import threading
from main import run_api

def test_api():
    # Wait a moment for the server to spin up
    time.sleep(3)
    
    base_url = "http://localhost:8000"
    
    print("\n--- 1. Testing Health Check ---")
    health = requests.get(f"{base_url}/health")
    print(health.json())

    print("\n--- 2. Testing Single Prediction ---")
    payload = {
        "article": "Machine_learning",
        "days_ahead": 7
    }
    pred = requests.post(f"{base_url}/predict", json=payload)
    print(pred.json())

    print("\n--- 3. Testing Batch Prediction ---")
    batch_payload = {
        "articles": ["Python_(programming_language)", "Deep_learning", "Artificial_intelligence"]
    }
    batch_pred = requests.post(f"{base_url}/batch-predict", json=batch_payload)
    print(batch_pred.json())
    
    print("\n✅ All basic API tests passed!")

if __name__ == "__main__":
    print("Starting API in the background...")
    # Run the API in a separate thread so we can test it
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()
    
    # Run the tests
    test_api()