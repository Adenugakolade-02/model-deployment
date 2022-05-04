import os
import sys

sys.path.append(os.getcwd())

from ml_api.api.app import create_app
from ml_api.api.config import DevelopmentConfig

appliction  = create_app(config_object=DevelopmentConfig)

if __name__=="__main__":
    appliction.run()