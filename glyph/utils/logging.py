import os
import sys
import logging
import logging.config
import yaml

# module name is in conflict with stdlib and can cause unwanted monkey patching


def load_config(config_file, placeholders={}, default_level=logging.INFO):
    """Load logging configuration from .yaml file."""
    if not sys.warnoptions:
        # Route warnings through python logging
        logging.captureWarnings(True)
    if os.path.exists(config_file):
        with open(config_file, 'rt') as f:
            content = f.read().format(**placeholders)
        config = yaml.load(content)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


def log_level(verbosity):
    """Convert numeric verbosity to logging log levels."""
    level = [logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]
    verbosity = len(level) - 1 if verbosity >= len(level) else verbosity
    return level[verbosity]
