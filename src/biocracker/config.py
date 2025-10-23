"""Module contains configuration settings for BioCracker."""

import logging
import os

LOGGER_NAME = "biocracker"
LOGGER_LEVEL = os.getenv("BIOCRACKER_LOG_LEVEL", "INFO").upper()
NAME_CACHE_DIR = os.getenv("NAME_CACHE_DIR", "biocracker_cache")
PARAS_CACHE_DIR_NAME = os.getenv("PARAS_CACHE_DIR_NAME", "paras_cache")
PARAS_MODEL_DOWNLOAD_URL = "https://zenodo.org/records/17224548/files/all_substrates_model.paras.gz?download=1"


# Setup logger
logging.basicConfig()
logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(LOGGER_LEVEL)
