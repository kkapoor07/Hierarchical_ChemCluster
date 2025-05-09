# src/common/timing_utils.py
import time
from functools import wraps
import logging

# Get logger for this module
logger = logging.getLogger(__name__)

def timing(f=None, message_prefix=""):
    """
    Decorator or function wrapper to measure execution time.

    Logs the execution time using the standard logging framework.

    Can be used as:
        @timing
        def my_func(): ...

        @timing(message_prefix="Preprocessing Step")
        def another_func(): ...

        # Or called directly (less common)
        # timed_func = timing(my_func, message_prefix="Direct Call")
        # timed_func(args)

    Args:
        f (callable, optional): The function to wrap. Used when applied as @timing.
        message_prefix (str, optional): A prefix to add to the log message.

    Returns:
        callable: The wrapped function.
    """
    if f is None: # Factory mode: allows usage like @timing(message_prefix="...")
        def decorator(func):
            return timing(func, message_prefix=message_prefix)
        return decorator

    @wraps(f)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter() # Use perf_counter for high resolution
        try:
            result = f(*args, **kwargs)
            success = True
        except Exception as e:
            success = False
            result = None # Or re-raise exception if needed
            logger.error(f"Exception during execution of '{f.__name__}': {e}", exc_info=True) # Log exception info
            raise # Re-raise the exception after logging
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time
            prefix = f"{message_prefix}: " if message_prefix else ""
            status = "completed" if success else "failed"
            log_message = f"{prefix}'{f.__name__}' {status} in {duration:.4f} seconds"
            if success:
                logger.info(log_message)
            else:
                logger.error(log_message) # Log failure time as error

        return result
    return wrapper
