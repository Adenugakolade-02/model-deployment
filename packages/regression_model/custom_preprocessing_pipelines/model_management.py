import logging
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from packages.regression_model.configuration import config
from packages.regression_model import __version__ as _version

_logger = logging.getLogger(__name__)



def load_data(file_name:str) -> pd.DataFrame:
    data = pd.read_csv(file_name)
    return data

def save_model(fitted_model:Pipeline) -> None:
    model_name = f"{config.PIPELINE_SAVE_FILE}{_version}.pkl"
    model_path = config.TRAINED_MODEL_DIR/model_name
    clear_stale_models(files_to_keep=model_name)
    joblib.dump(fitted_model,model_path)

    _logger.info(
        f"Saved Model {model_name}"
    )

def load_model(filename:str) -> Pipeline:
    model_path = config.TRAINED_MODEL_DIR/filename
    loaded_model = joblib.load(filename=model_path)
    return loaded_model

def clear_stale_models(files_to_keep)-> None:
    for files in config.TRAINED_MODEL_DIR.iterdir():
        if files not in [files_to_keep, "__init__.py"]:
            _logger.info(
                f"Files has been deleted successfully {files}"
            )
            files.unlink()