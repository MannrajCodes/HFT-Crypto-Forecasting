import json
import time
import logging
import requests
from typing import Dict, Any, Optional
from kafka import KafkaProducer
from config.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeCollector:
    def __init__(self, topic: str = settings.kafka_topic):
        self.topic = topic
        self.article = settings.default_article
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=settings.kafka_broker,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                retries=5
            )
            logger.info(f"Connected to Kafka Broker at {settings.kafka_broker}")
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            self.producer = None

    def fetch_data_with_backoff(self, endpoint: str, max_retries: int = 3) -> Optional[Dict[str, Any]]:
        # ROLLBACK: The 24hr Ticker. Smoother, highly predictable wave.
        live_endpoint = "https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT"
        
        try:
            response = requests.get(live_endpoint, timeout=3)
            response.raise_for_status()
            live_data = response.json()
            
            # Target: Live executed Bitcoin Price
            current_price = float(live_data['lastPrice'])
            
            # Feature: Number of trades (Overall market momentum)
            trade_count = float(live_data['count'])

            return {
                "timestamp": time.time(),
                "article": self.article, 
                "views": current_price,      
                "unique_visitors": trade_count
            }
            
        except Exception as e:
            logger.error(f"Binance API Error: {e}")
            return None

    def collect_and_stream(self) -> Optional[Dict[str, Any]]:
        if not self.producer: return None
        data = self.fetch_data_with_backoff(endpoint="binance")
        if data:
            self.producer.send(self.topic, value=data)
            self.producer.flush()
            return data
        return None

collector = RealTimeCollector()