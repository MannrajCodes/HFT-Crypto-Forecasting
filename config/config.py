import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "HFT Crypto Forecasting Engine"
    version: str = "3.0.0-quant"
    debug: bool = True
    port: int = 8000
    host: str = "0.0.0.0"

    # Kafka Settings
    kafka_broker: str = os.getenv("KAFKA_BROKER", "localhost:9092")
    kafka_topic: str = "crypto_stream"

    # InfluxDB Settings (Kept the same so Grafana DB connection doesn't break)
    influx_url: str = os.getenv("INFLUX_URL", "http://localhost:8086")
    influx_token: str = os.getenv("INFLUX_TOKEN", "my-super-secret-auth-token")
    influx_org: str = os.getenv("INFLUX_ORG", "forecast_org")
    influx_bucket: str = "forecast_bucket" 

    # Quant Trading Hyperparameters
    sequence_length: int = 64      # SWT Wavelet optimal memory
    forecast_horizon: int = 1      # Predicting the IMMEDIATE next market tick!
    features: int = 2              # Target (Price) + Feature (Trade Volume)
    learning_rate: float = 0.0002  # Steady learning for financial data

    # Default asset
    default_article: str = "BTC-USDT"

settings = Settings()