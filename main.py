import argparse
import uvicorn
import logging
from config.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_api():
    logger.info(f"Starting API Server on port {settings.port}...")
    uvicorn.run("api.model_server:app", host=settings.host, port=settings.port, reload=settings.debug)

def run_collector(article: str):
    from data_fetcher.realtime_collector import collector
    import time
    
    logger.info(f"Starting real-time collector for: {article}")
    collector.article = article
    try:
        while True:
            data = collector.collect_and_stream()
            if data:
                logger.info(f"Collected: {data}")
            time.sleep(2)
    except KeyboardInterrupt:
        logger.info("Collector stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"{settings.app_name} CLI")
    parser.add_argument("mode", choices=["api", "collector"], help="Which component to run")
    parser.add_argument("--article", type=str, default=settings.default_article)
    
    args = parser.parse_args()

    if args.mode == "api":
        run_api()
    elif args.mode == "collector":
        run_collector(args.article)