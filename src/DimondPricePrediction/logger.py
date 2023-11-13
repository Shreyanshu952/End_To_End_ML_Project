import logging
import os
from datetime import datetime

# creating the format of the name of the logging file
LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# creating a "logs" folder inside the current working directory
log_path=os.path.join(os.getcwd(), "logs")

os.makedirs(log_path, exist_ok=True)

# joining paths of the logging file with "logs" folder
LOG_FILE_PATH = os.path.join(log_path, LOG_FILE)

# setting the basic configuration for the logging
logging.basicConfig(
    level=logging.INFO,
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] - %(lineno)d - %(name)s - %(levelname)s - %(message)s"
)

if __name__=="__main__":
    logging.info("Here again i am testing for the fourth time")