from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler

from packages.regression_model.regression_model.configuration import config
from packages.regression_model.regression_model.custom_preprocessing_pipelines import preprocessing as pp

price_pipeline = Pipeline([
    (
        'Categorical Imputer',pp.CategoricalImputer(variables=config.CATEGORICAL_VARS_WITH_NA)
    ),
    (
        'Numerical Imputer',pp.NumericalImputer(variables=config.NUMERICAL_VARS_WITH_NA)
    ),
    (
        'Temporal Variable',pp.TemporalVariableEstimator(variables=config.TEMPORAL_VARS,reference_variables=config.DROP_FEATURE)
    ),
    (
        'Log Transformer',pp.LogTransFormer(variables=config.NUMERICALS_LOG_VARS)
    ),
    (
        'Rare Encoder',pp.RareLabelEncoder(variables=config.CATEGORICAL_VARS, tolerance=0.01)        
    ),
    (
        'Encode Categories',pp.CategoricalEncoder(variables=config.CATEGORICAL_VARS)
    ),
    (
        'Drop Features',pp.DropUnecessaryFeatures(variables=config.DROP_FEATURE)
    ),
    (
        'Scaler',MinMaxScaler()
    ),
    (
        'Linear Model',Lasso(alpha=0.05,random_state=0)
    )
])


