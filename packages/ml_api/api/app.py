from flask import Flask
from packages.ml_api.api.controller import prediction_app
from packages.ml_api.api.controller import get_logger


_logger = get_logger(__name__)

def create_app(*,config_object) -> Flask:
    """Creating an instance of the flask app"""

    flask_app  = Flask(import_name="ml_api")
    flask_app.config.from_object(config_object)
    
    flask_app.register_blueprint(prediction_app)
    _logger.debug("Application interface created")
    return flask_app

