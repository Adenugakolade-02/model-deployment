import pandas as pd
import numpy as np

#creating the scikit learn pipeline

#importing the neccesary features to create the pipelines
from sklearn.base import BaseEstimator, TransformerMixin


class CategoricalImputer(BaseEstimator, TransformerMixin):
    """Filling NAN categorical variables with 'Missing' """
    def __init__(self, variables=None):
        if not isinstance(variables,list):
            self.variables = [variables]
        else:
            self.variables = variables
    
    def fit(self,X:pd.DataFrame, y:pd.Series):
        return self
    
    def transform(self,X:pd.DataFrame):
        X = X.copy()
        for features in self.variables:
            X[features] = X[features].fillna('Missing')
        return X

    
class NumericalImputer(BaseEstimator, TransformerMixin):
    """For filling missing numerical features with its  Mode"""
    def __init__(self, variables=None):
        if not isinstance(variables,list):
            self.variables = variables
        else:
            self.variables = variables
    
    def fit(self, X:pd.DataFrame, y:pd.Series=None):
        self.impute_dictionary = {}
        for features in self.variables:
            self.impute_dictionary[features] = X[features].mode()[0]
        return self

    def transform(self, X:pd.DataFrame):
        X = X.copy()
        for features in self.variables:
            X[features].fillna(self.impute_dictionary[features],inplace=True)
        return X

class TemporalVariableEstimator(BaseEstimator, TransformerMixin):
    """Extracts info from time variables"""
    def __init__(self, variables, reference_variables):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
        self.reference_variables = reference_variables

    
    def fit(self, X:pd.DataFrame,y:pd.Series=None):
        return self
    
    def transform(self, X:pd.DataFrame, y:pd.Series=None):
        X = X.copy()
        for features in self.variables:
            X[features] = X[self.reference_variables] - X[features]
        return X


class RareLabelEncoder(BaseEstimator, TransformerMixin):
    """Filling rare categories in feature with 'Rare' """
    def __init__(self,variables, tolerance):
        self.tolerance = tolerance
        if not isinstance(variables,list):
            self.variables = [variables]
        else:
            self.variables = variables
    
    def fit(self, X:pd.DataFrame, y:pd.Series=None):
        self.rare_encoder = {}
        
        for features in self.variables:
            value_list  = X[features].value_counts()/len(X)
            self.rare_encoder[features] = list(value_list[value_list>self.tolerance].index)
        return self
    def transform(self, X:pd.DataFrame):
        X = X.copy()
        for feature in self.variables:
            X[feature] = np.where(X[feature].isin(self.rare_encoder[feature]), X[feature], 'Rare')
        return X

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encoding Categorical variables"""
    def __init__(self,variables):
        if not isinstance(variables,list):
            self.variables = [variables]
        else:
            self.variables = variables
    
    def fit(self, X:pd.DataFrame, y:pd.Series=None):
        new_frame = pd.concat([X,y],axis=1)
        new_frame.columns = X.columns.tolist() + ["target"]
        self.ordered_map = {}
        
        for feature in self.variables:
            group = new_frame.groupby([feature])["target"].mean().sort_values(ascending=True).index
            self.ordered_map[feature] = {k:i for i,k in enumerate(group,0)}
        return self
    
    def transform(self, X:pd.DataFrame):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.ordered_map[feature])

        ###Checking for nan values were introduced
        if X[self.variables].isnull().any().any():
            null_items = X[self.variables].isnull().any().items()
            null_dictionary = {key:value for (key, value) in null_items if value is True}

            raise ValueError(
                f"Categorical encoder has introduced NaN when "
                f"transforming categorical variables: {null_dictionary.keys()}"
            )
        return X



class LogTransFormer(BaseEstimator, TransformerMixin):
    def __init__(self,variables):
        if not isinstance(variables,list):
            self.variables = [variables]
        else:
            self.variables = variables
    
    def fit(self, X:pd.DataFrame, y:pd.Series=None):
        return self
    
    def transform(self,X:pd.DataFrame):
        X = X.copy()

        if not (X[self.variables]>0).all().all():
            error_features = [(X[self.variables]<=0).any().index]

            raise ValueError(
            f"""Variable contains Zero or negative value
            cannot apply Log transform to features {error_features}
            """
            )
        for feature in self.variables:
            X[feature]  = np.log(X[feature])
        return X
class DropUnecessaryFeatures(BaseEstimator, TransformerMixin):
    def __init__(self,variables):
        if not isinstance(variables,list):
            self.variables = variables
        else:
            self.variables = variables
    def fit(self,X:pd.DataFrame, y:pd.Series):
        return self

    def transform(self,X:pd.DataFrame):
        X = X.copy()
        X.drop(self.variables,axis=1 ,inplace=True)
        return X