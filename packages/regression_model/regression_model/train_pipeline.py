import logging

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


from packages.regression_model.regression_model.configuration import config
from packages.regression_model.regression_model.custom_preprocessing_pipelines import model_management
import packages.regression_model.regression_model.pipeline as pipeline
from packages.regression_model.regression_model import __version__ as _version




_logger = logging.getLogger(__name__)

def train_model() -> None:
    ###reading the data
    data = model_management.load_data(file_name=config.TRAINING_DATA)
    X_train, X_test,y_train,y_test = train_test_split(data[config.FEATURES],data[config.TARGET],test_size=0.2,random_state=42)

    y_train = np.log(y_train)
    pipeline.price_pipeline.fit(X_train[config.FEATURES],y_train)

    _logger.info(f"Saving model version: {_version}")
    model_management.save_model(pipeline.price_pipeline)





if __name__ == "__main__":
    train_model()