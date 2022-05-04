from itertools import compress

import pandas as pd
import numpy as np
from packages.regression_model.configuration import config


def validate_dataset(input_data:pd.DataFrame) -> pd.DataFrame:
    validated_data = input_data.copy()
    # validated_data.dropna(axis=0,inplace=True)
    validated_data = validated_data[config.FEATURES]
    validated_data.dropna(axis=0,inplace=True,subset=config.NUMERICALS_LOG_VARS)

    # if input_data[config.CATEGORICAL_NA_NOT_ALLOWED].isnull().any().any():
    #     validated_data = validated_data.dropna(axis=0, subset=config.CATEGORICAL_NA_NOT_ALLOWED)
    
    # if input_data[config.NUMERICAL_VARS_WITH_NA].isnull().any().any():
    #     validated_data = validated_data.dropna(axis=0, subset=config.NUMERICAL_VARS_WITH_NA)

    # if input_data[config.NUMERICALS_LOG_VARS].isnull().any().any():
    #     validated_data = validated_data.dropna(axis=0,subset=config.NUMERICALS_LOG_VARS)
        # na_fetures = compress(config.NUMERICALS_LOG_VARS, (input_data[config.NUMERICALS_LOG_VARS] <= 0).any().values.tolist())
        # var_features_with_na = [x for x in na_fetures]
        # print(f'Here are the numerical values with nan {var_features_with_na}')

        # for features in var_features_with_na:
        #     validated_data = validated_data[validated_data[features]>0]
        #     # validated_data.replace([np.inf, -np.inf], np.nan)
        #     validated_data.dropna(axis=0,subset=[features],inplace=True)
            
        # # validated_data = validated_data[validated_data[var_features_with_na]>0]
    return validated_data
