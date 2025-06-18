import os
import random
from prefect import flow, task
from prefect.logging import get_run_logger

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PREFECT_API_URL", "http://127.0.0.1:4200/api")

@task(retries=2, retry_delay_seconds=1)
def check_random():
    logger = get_run_logger()
    value = random.random()
    logger.info(f"Random draw: {value:.3f}")
    if value < 0.5:
        logger.warning("Low model performance: retraining triggered.")
        raise ValueError("Model requires retraining")
    else:
        logger.info("Model performance is stable. No action needed.")

@flow
def periodic_check():
    check_random()

if __name__ == "__main__":
    periodic_check.serve(
        name="every-30s",
        interval=30
    )