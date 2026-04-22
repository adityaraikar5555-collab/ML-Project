from src.logger import logging
from src.exception import CustomException
import sys

def test_function():
    try:
        x = 10 / 0   # intentional error
    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        logging.info("Test started")
        test_function()
    except Exception as e:
        logging.error(e)