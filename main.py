"""Entry point: initialise the platform and start the Flask chatbot server."""

import os
import logging

# Load .env file if present (before importing anything that reads env vars)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from pipeline.config import ALL_DATA_DIRS, FLASK_PORT
from chatbot.job_manager import init_db
from chatbot.app import app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    # Ensure all data directories exist
    for d in ALL_DATA_DIRS:
        os.makedirs(d, exist_ok=True)
        logger.debug("Data dir ready: %s", d)

    # Initialise SQLite job database
    init_db()
    logger.info("Job database initialised.")

    # Start Flask
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    logger.info("Starting LINE chatbot server on port %d …", FLASK_PORT)
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=debug)


if __name__ == "__main__":
    main()
