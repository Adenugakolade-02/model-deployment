import logging
import pandas as pd
import numpy as np


from packages.regression_model.regression_model.custom_preprocessing_pipelines.model_management import load_model
from packages.regression_model.regression_model.custom_preprocessing_pipelines import validation
from packages.regression_model.regression_model.configuration import config

from packages.regression_model import __version__ as _version

_logger = logging.getLogger(__name__)

# pipeline_filename = "regression_model.pkl"
pipeline_filename = f"{config.PIPELINE_SAVE_FILE}{_version}.pkl"

trained_model  = load_model(filename=pipeline_filename)

def make_prediction(input_data) -> dict:
    data = pd.read_json(input_data)
    validated_data = validation.validate_dataset(input_data=data)
    prediction = trained_model.predict(validated_data[config.FEATURES])
    output  = np.exp(prediction)
    response = {'prediction':output}

    _logger.info(
        f"Making prdcictions with model versions: {_version}"
        f"Inputs: {validated_data}"
        f"predictions: {response}"
        )
        
    return response

