import pathlib
import packages.regression_model.regression_model as regression_model

from packages.regression_model.regression_model.custom_preprocessing_pipelines import preprocessing

PACKAGE_ROOT = pathlib.Path(regression_model.__file__).resolve().parent
# print(PACKAGE_ROOT)
DATASET_DIR = PACKAGE_ROOT/'Datasets'
TRAINED_MODEL_DIR = PACKAGE_ROOT/'trained_models'

###Data sets
TRAINING_DATA = DATASET_DIR/"train.csv"
TESTING_DATA = DATASET_DIR/"test.csv"
TARGET = "SalePrice"

FEATURES = ['MSSubClass', 'MSZoning', 'Neighborhood', 'OverallQual', 'OverallCond',
       'YearRemodAdd', 'RoofStyle', 'MasVnrType', 'BsmtQual', 'BsmtExposure',
       'HeatingQC', 'CentralAir', '1stFlrSF', 'GrLivArea', 'BsmtFullBath',
       'KitchenQual', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageFinish', 'GarageCars', 'PavedDrive', 'LotFrontage' ,'YrSold']

###Features to be dropped
DROP_FEATURE = 'YrSold'

###Numerical features with NA
NUMERICAL_VARS_WITH_NA = ['LotFrontage']

# categorical variables with NA in train set
CATEGORICAL_VARS_WITH_NA = [
    "MasVnrType",
    "BsmtQual",
    "BsmtExposure",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
]

TEMPORAL_VARS = "YearRemodAdd"

# variables to log transform
NUMERICALS_LOG_VARS = ["LotFrontage", "1stFlrSF", "GrLivArea"]

# categorical variables to encode
CATEGORICAL_VARS = [
    "MSZoning",
    "Neighborhood",
    "RoofStyle",
    "MasVnrType",
    "BsmtQual",
    "BsmtExposure",
    "HeatingQC",
    "CentralAir",
    "KitchenQual",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "PavedDrive",
]

NUMERICAL_NA_NOT_ALLOWED = [
    feature
    for feature in FEATURES
    if feature not in CATEGORICAL_VARS + NUMERICAL_VARS_WITH_NA
]

CATEGORICAL_NA_NOT_ALLOWED = [
    feature for feature in CATEGORICAL_VARS if feature not in CATEGORICAL_VARS_WITH_NA
]

PIPELINE_NAME = "lasso regression"
PIPELINE_SAVE_FILE = F"{PIPELINE_NAME}_output_v"