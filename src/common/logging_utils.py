# src/common/logging_utils.py
import logging
import sys
import os
from logging.handlers import RotatingFileHandler

def setup_logging(log_file_path=None, level=logging.INFO, include_console=True, max_bytes=10*1024*1024, backup_count=5):
    """
    Configures logging to file (with rotation) and/or console.

    Removes existing handlers from the root logger before adding new ones
    to prevent duplicate messages if called multiple times.

    Args:
        log_file_path (str, optional): Path to the log file. If None, logs only to console.
                                       Directory will be created if it doesn't exist.
        level (int, optional): Logging level (e.g., logging.INFO, logging.DEBUG).
                               Defaults to logging.INFO.
        include_console (bool, optional): Whether to also log to the console (stderr).
                                          Defaults to True.
        max_bytes (int, optional): Maximum size of the log file before rotation.
                                   Defaults to 10MB.
        backup_count (int, optional): Number of backup log files to keep.
                                      Defaults to 5.
    """
    log_formatter = logging.Formatter(
        "%(asctime)s [%(process)d:%(threadName)s] [%(levelname)-5.5s] [%(name)s] %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    root_logger = logging.getLogger()

    # Set level first
    root_logger.setLevel(level)

    # Remove existing handlers from the root logger
    # This prevents duplication if setup_logging is called multiple times
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close() # Close handler before removing

    # File Handler (with rotation)
    if log_file_path:
        try:
            # Create directory if it doesn't exist
            log_dir = os.path.dirname(log_file_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
                print(f"Created log directory: {log_dir}") # Print confirmation as logging might not be fully set up

            # Use RotatingFileHandler
            file_handler = RotatingFileHandler(
                log_file_path, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8'
            )
            file_handler.setFormatter(log_formatter)
            root_logger.addHandler(file_handler)
            root_logger.info(f"Logging initialized. Log file: {log_file_path}")

        except Exception as e:
            # Fallback to console if file logging fails
            print(f"Error setting up file logger at {log_file_path}: {e}. Falling back to console logging.", file=sys.stderr)
            include_console = True # Ensure console logging is active
            log_file_path = None # Mark file logging as failed

    # Console Handler
    if include_console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(log_formatter)
        root_logger.addHandler(console_handler)

    if not log_file_path and not include_console:
        # Add NullHandler to prevent "No handlers could be found" warning
        # if no logging destination is configured.
        root_logger.addHandler(logging.NullHandler())
        # Optionally log a warning if possible (though no handler might see it)
        # root_logger.warning("Logging setup called with no file path and no console output.")
        print("Warning: Logging setup called with no file path and no console output.", file=sys.stderr)
