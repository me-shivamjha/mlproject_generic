import logging 
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"  # Create a log file name based on the current date and time.
LOG_PATH = os.path.join(os.getcwd(), "logs")  # Define the path to the logs directory.
os.makedirs(LOG_PATH, exist_ok=True)  # Create the "logs" directory if it doesn't exist.

LOG_FILE_PATH = os.path.join(LOG_PATH, LOG_FILE)  # Define the full path to the log file.

logging.basicConfig(
    filename=LOG_FILE_PATH,  # Set the log file path for logging output.
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",  # Define the log message format to include timestamp, logger name, log level, and message.
    level=logging.INFO  # Set the logging level to INFO, which means that only messages with a severity of INFO or higher will be logged.
)

if __name__ == "__main__":
    logging.info("Logging has been configured successfully.")  # Log an informational message indicating that logging has been set up successfully.