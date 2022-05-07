from itertools import compress

import pandas as pd
import numpy as np
import logging

from packages.regression_model.regression_model.configuration import config

_logger = logging.getLogger(__name__)


def validate_dataset(input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for unprocessable values."""

    # print(f"This the lenght of the data that  entered {len(input_data)}")
    _logger.info(f"This the lenght of the data that  entered {len(input_data)}")

    validated_data = input_data.copy()

    # check for numerical variables with NA not seen during training
    if input_data[config.NUMERICAL_NA_NOT_ALLOWED].isnull().any().any():
        validated_data = validated_data.dropna(
            axis=0, subset=config.NUMERICAL_NA_NOT_ALLOWED
        )

    # check for categorical variables with NA not seen during training
    if input_data[config.CATEGORICAL_NA_NOT_ALLOWED].isnull().any().any():
        validated_data = validated_data.dropna(
            axis=0, subset=config.CATEGORICAL_NA_NOT_ALLOWED
        )

    # check for values <= 0 for the log transformed variables
    if (input_data[config.NUMERICALS_LOG_VARS] <= 0).any().any():
        vars_with_neg_values = config.NUMERICALS_LOG_VARS[
            (input_data[config.NUMERICALS_LOG_VARS] <= 0).any()
        ]
        validated_data = validated_data[validated_data[vars_with_neg_values] > 0]

    # print(f"This the lenght of the data leaving {len(validated_data)}")
    _logger.info(f"This the lenght of the data that  entered {len(validated_data)}")
    return validated_data