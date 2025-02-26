import time
import logging


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def log_time(func: callable) -> callable:
    def wrapper(*args, **kwargs):
        logging.info(f"Running {func.__name__}...")
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        logging.info(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper