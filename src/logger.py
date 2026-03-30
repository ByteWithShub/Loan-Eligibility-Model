#Creates a logger that writes to a file in the logs directory. 
#The log file is named app.log and includes timestamps, log levels, and messages. 
#The logs directory is created if it does not exist.

import logging
import os


def setup_logger():
    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        filename="logs/app.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    return logging.getLogger(__name__)