import json
import asyncio
from typing import List, Optional
from fastapi import FastAPI, BackgroundTasks, WebSocket, HTTPException, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from config.config import settings

# Initialize FastAPI App
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Production-ready real-time forecasting API with WebSocket support."
)
Instrumentator().instrument(app).expose(app)
# Enable CORS for frontend/dashboard integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for Request Validation ---
class PredictRequest(BaseModel):
    article: str = settings.default_article
    days_ahead: int = settings.forecast_horizon

class BatchPredictRequest(BaseModel):
    articles: List[str]

# --- REST Endpoints ---
@app.get("/")
async def root():
    return {"message": f"Welcome to {settings.app_name}", "version": settings.version}

@app.get("/health")
async def health_check():
    # In a full deployment, this checks connections to Redis, Kafka, and InfluxDB
    return {
        "status": "healthy",
        "model_architecture": "CNN-GRU Hybrid",
        "services": {
            "api": "online",
            "streaming_pipeline": "standby"
        }
    }

@app.post("/predict")
async def predict(request: PredictRequest):
    """Standard REST endpoint for single-article forecasting."""
    try:
        # TODO: Integrate with preprocessing.streaming_pipeline and models.improved_models
        # Mocking the response for the architecture phase
        mock_forecast = [100.5 + (i * 2.1) for i in range(request.days_ahead)]
        
        return {
            "article": request.article,
            "forecast": mock_forecast,
            "confidence_interval": "95%",
            "model_used": "hybrid_cnn_gru_v2"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-predict")
async def batch_predict(request: BatchPredictRequest):
    """Process multiple targets at once."""
    results = {}
    for article in request.articles:
        results[article] = {"status": "queued_for_processing"}
    return {"results": results}

@app.post("/model/update")
async def trigger_retraining(background_tasks: BackgroundTasks, trigger_retrain: bool = True):
    """Endpoint to manually trigger continuous learning."""
    if trigger_retrain:
        # background_tasks.add_task(online_trainer.train_incremental)
        return {"status": "Retraining job added to background tasks."}
    return {"status": "Ignored."}

# --- WebSocket Endpoints ---
@app.websocket("/ws/predictions")
async def websocket_predictions(websocket: WebSocket):
    """
    Real-time streaming endpoint. 
    Accepts JSON with an article name and streams back predictions continuously.
    """
    await websocket.accept()
    try:
        while True:
            # Wait for client to request a specific stream
            data = await websocket.receive_text()
            request_data = json.loads(data)
            article = request_data.get("article", settings.default_article)
            
            await websocket.send_json({
                "type": "info", 
                "message": f"Subscribed to real-time stream for: {article}"
            })

            # Simulate listening to a Kafka stream/Continuous predictions
            for _ in range(5):  # Limiting loop for demonstration
                await asyncio.sleep(2) # Simulate processing delay
                await websocket.send_json({
                    "type": "prediction_update",
                    "article": article,
                    "real_time_value": 150.2,
                    "anomaly_detected": False
                })
                
    except WebSocketDisconnect:
        print("Client disconnected from WebSocket.")
    except Exception as e:
        print(f"WebSocket error: {e}")