# src/common/utils.py
import yaml
import argparse
import logging
import os

logger = logging.getLogger(__name__)

def load_config_and_reparse(parser: argparse.ArgumentParser, args_initial: argparse.Namespace) -> argparse.Namespace:
    """
    Loads configuration from a YAML file specified by args_initial.config,
    sets parser defaults from the loaded config, and then re-parses the
    command line arguments using the updated defaults.

    This ensures that command-line arguments always override values specified
    in the configuration file, which in turn override the hardcoded defaults
    in ArgumentParser.add_argument().

    Args:
        parser (argparse.ArgumentParser): The ArgumentParser object for the script.
        args_initial (argparse.Namespace): Namespace object from an initial parse
                                           (usually using parse_known_args())
                                           to get the config file path.

    Returns:
        argparse.Namespace: The final Namespace object with arguments resolved
                           according to the priority CLI > config > hardcoded defaults.

    Raises:
        SystemExit: If the config file is specified but not found or cannot be parsed,
                    it calls parser.error(), which typically exits.
    """
    config_path = getattr(args_initial, 'config', None)
    config = {}

    if config_path:
        # Check if the file exists before attempting to open
        if not os.path.isfile(config_path): # More specific check than os.path.exists
            parser.error(f"Configuration file specified but not found: {config_path}")

        logger.info(f"Loading configuration from: {config_path}")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                if config is None: # Handle empty YAML file case
                    config = {}
                    logger.warning(f"Configuration file {config_path} is empty.")
        except yaml.YAMLError as e:
            # Use parser.error for clean exit with informative message
            parser.error(f"Error parsing configuration file {config_path}: {e}")
        except Exception as e:
             # Catch other potential file reading errors
             parser.error(f"Error reading configuration file {config_path}: {e}")

    else: # No --config argument provided
        logger.info("No configuration file specified. Using command-line arguments and defaults.")


    # Set parser defaults from the loaded configuration dictionary
    # Only update defaults for keys present in the config file
    defaults_to_set = {k: v for k, v in config.items() if hasattr(argparse.Namespace, k) or k in parser._actions}
    # A stricter check might involve iterating parser._actions more carefully
    if defaults_to_set:
         logger.debug(f"Updating parser defaults from config: {defaults_to_set}")
         parser.set_defaults(**defaults_to_set)
    else:
         logger.debug("Config file was empty or contained no relevant keys for parser defaults.")


    # Re-parse the command-line arguments using the updated defaults
    # sys.argv[1:] contains the original command line arguments passed to the script
    args_final = parser.parse_args() # This re-parses sys.argv

    # Log the final, effective arguments
    logger.info("Effective arguments after processing CLI and config file:")
    # Sort arguments for consistent logging order
    for key, value in sorted(vars(args_final).items()):
        logger.info(f"  {key}: {value}")

    return args_final
