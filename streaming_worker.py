import json
import logging
import numpy as np
from collections import deque
from kafka import KafkaConsumer
from config.config import settings
from preprocessing.streaming_pipeline import pipeline
from models.improved_models import forecast_model
from training.online_trainer import online_trainer as brain_instance
from database.influx_connector import influx_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# The waiting room for inputs
training_queue = deque(maxlen=settings.forecast_horizon)

def run_worker():
    logger.info("🚀 Starting Quant Trading Streaming Worker...")
    try:
        consumer = KafkaConsumer(
            settings.kafka_topic,
            bootstrap_servers=settings.kafka_broker,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest'
        )
    except Exception as e:
        logger.error(f"❌ Could not connect to Kafka: {e}")
        return

    for message in consumer:
        data = message.value
        article = data.get('article', 'BTC-USDT')
        actual_price = float(data.get('views', 0))
        volume = float(data.get('unique_visitors', 0))

        # Save actuals to InfluxDB
        influx_db.save_actual(article, actual_price, unique_visitors=volume)

        # Process through SWT Pipeline
        is_anomaly, model_input = pipeline.ingest_new_data(data)

        if model_input is not None:
            # Predict the next price tick
            predictions = forecast_model.predict(model_input)
            future_pred_scaled = predictions[0][0] # Get the 1st step
            
            dummy_array = np.zeros((1, pipeline.feature_count))
            dummy_array[0, 0] = future_pred_scaled
            predicted_price = float(pipeline.scaler.inverse_transform(dummy_array)[0, 0])
            
            influx_db.save_prediction(article, predicted_price, horizon_step=settings.forecast_horizon)
            logger.info(f"📈 [BTC-USDT] Actual: ${actual_price:.2f} | Forecast: ${predicted_price:.2f}")

            # --- THE BUG FIX: Correct Target Array ---
            training_queue.append(model_input[0])
            
            if len(training_queue) == settings.forecast_horizon:
                old_input = training_queue.popleft() # Pop the oldest input
                
                dummy_target = np.zeros((1, pipeline.feature_count))
                dummy_target[0, 0] = actual_price
                scaled_true_future = pipeline.scaler.transform(dummy_target)[0, 0]
                
                # We now train on the exact scalar value, NOT a flat array line!
                target_array = np.array([scaled_true_future]) 
                brain_instance.add_sample(old_input, target_array)

        else:
            logger.info(f"⏳ Buffering market data... ({len(pipeline.data_buffer)}/{pipeline.window_size})")

if __name__ == "__main__":
    run_worker()