from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import logging
from config.config import settings

logger = logging.getLogger(__name__)

class InfluxConnector:
    def __init__(self):
        self.client = InfluxDBClient(
            url=settings.influx_url,
            token=settings.influx_token,
            org=settings.influx_org
        )
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.bucket = settings.influx_bucket

    def save_actual(self, article: str, views: float, unique_visitors: float):
        point = Point("actuals") \
            .tag("article", article) \
            .field("true_value", float(views)) \
            .field("unique_visitors", float(unique_visitors))
        try:
            self.write_api.write(bucket=self.bucket, org=settings.influx_org, record=point)
        except Exception as e:
            logger.error(f"Failed to write actuals to InfluxDB: {e}")

    def save_prediction(self, article: str, predicted_value: float, horizon_step: int):
        point = Point("forecasts") \
            .tag("article", article) \
            .tag("horizon_step", str(horizon_step)) \
            .tag("model_version", settings.version) \
            .field("predicted_value", float(predicted_value))
        try:
            self.write_api.write(bucket=self.bucket, org=settings.influx_org, record=point)
        except Exception as e:
            logger.error(f"Failed to write prediction to InfluxDB: {e}")

influx_db = InfluxConnector()