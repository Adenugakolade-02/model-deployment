import logging
import sys

from packages.regression_model.configuration import config
from packages.regression_model.configuration import logging_config

formatter = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s —" "%(funcName)s:%(lineno)d — %(message)s")
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

VERSION_PATH =  config.PACKAGE_ROOT/'VERSION'
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(console_handler)



with open(VERSION_PATH,'r') as version_file:
    __version__ = version_file.read().strip()