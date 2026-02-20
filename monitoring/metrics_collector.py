import time
from functools import wraps
from prometheus_client import Counter, Histogram, Gauge

# --- Define Prometheus Metrics ---

# 1. API Metrics
REQUEST_COUNT = Counter(
    'api_request_total', 
    'Total number of requests received', 
    ['endpoint', 'method', 'status']
)

REQUEST_LATENCY = Histogram(
    'api_request_latency_seconds', 
    'Request latency in seconds',
    ['endpoint']
)

# 2. Model Metrics
PREDICTION_VALUE = Gauge(
    'model_prediction_value', 
    'Latest predicted value from the CNN-GRU model',
    ['article']
)

ANOMALY_COUNT = Counter(
    'model_anomalies_detected_total', 
    'Total number of anomalies flagged by the streaming pipeline',
    ['article']
)

# --- Decorators for FastAPI ---

def track_metrics(endpoint_name: str):
    """
    A decorator to wrap around FastAPI endpoints to automatically 
    calculate latency and count requests.
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise e
            finally:
                duration = time.time() - start_time
                REQUEST_LATENCY.labels(endpoint=endpoint_name).observe(duration)
                REQUEST_COUNT.labels(
                    endpoint=endpoint_name, 
                    method="POST/GET", 
                    status=status
                ).inc()
                
        # Handle synchronous functions as well
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise e
            finally:
                duration = time.time() - start_time
                REQUEST_LATENCY.labels(endpoint=endpoint_name).observe(duration)
                REQUEST_COUNT.labels(
                    endpoint=endpoint_name, 
                    method="POST/GET", 
                    status=status
                ).inc()

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator