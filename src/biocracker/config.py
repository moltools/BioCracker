"""Module contains configuration settings for BioCracker."""

import os


LOGGER_NAME = "biocracker"
LOGGER_LEVEL = os.getenv("BIOCRACKER_LOG_LEVEL", "INFO").upper()
