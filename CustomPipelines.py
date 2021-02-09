import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

class DropFeatures(TransformerMixin, BaseEstimator):
    """
    Drops explicit columns from a pandas dataframe. 
    The columns to be dropped are given as a list of column names.
    """
    def __init__(self, feature_list:list):
        self._feature_list = feature_list

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame, y=None):
        X_copy = X.copy()
        for feature in self._feature_list:
            try:
                X_copy.drop(columns=[feature], inplace=True)
            except KeyError:
                print(f'DropFeatures warning: {feature} does not exist.')
        return X_copy


class CreateFeatures(TransformerMixin, BaseEstimator):
    """
    A transformer which augments new columns to a pandas dataframe. The new columns are 
    given as a dictionary with keys the column names and the values as pandas series objects.
    """
    def __init__(self, feature_dict:dict):
        self._feature_dict = feature_dict
    
    def fit(self, X:pd.DataFrame, y=None):
        return self

    def transform(self, X:pd.DataFrame, y=None):
        X_copy = X.copy()
        for feature in self._feature_dict.keys():
            X_copy[feature] = self._feature_dict[feature]
        return X_copy


class TransformFeatures(TransformerMixin, BaseEstimator):
    """
    A transformer which applies specified functions to specified columns of a 
    pandas DataFrame. The constructor inputs a dict of feature name, function pairs.
    """
    def __init(self, function_dict:dict):
        self._function_dict = funtion_dict

    def fit(self, X:pd.DataFrame, y=None):
        return self

    def transform(self, X:pd.DataFrame, y=None):
        X_copy = X.copy()
        for feature in self._function_dict.keys():
            try:
                X_copy[feature] = X_copy[feature].apply(self._function_dict[feature])
            except KeyError:
                print(f'TransformFeatures warning: {feature} does not exist.')
        return X_copy

