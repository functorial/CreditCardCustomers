import pandas as pd
import numpy as np
import CustomHelpers as chelp
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

class DropFeatures(BaseEstimator, TransformerMixin):
    """
    Drops explicit columns from a pandas dataframe. 
    The columns to be dropped are given as a list of column names.
    """
    def __init__(self, feature_list):
        self.feature_list = feature_list

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_ = X.copy()
        if self.feature_list is not None:
            for feature in self.feature_list:
                try:
                    X_.drop(columns=feature, inplace=True)
                except KeyError:
                    print(f'DropFeatures.transform Warning: {feature} does not exist.')

        self.feature_list_ = X_.columns

        return X_


class CreateFeatures(BaseEstimator, TransformerMixin):
    """
    A transformer which augments new columns to a pandas dataframe. The new columns are 
    given as a dictionary with keys the column names and the values as functions which 
    return pandas series objects.
    """
    def __init__(self, feature_dict):
        self.feature_dict = feature_dict
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        if self.feature_dict.keys() is not None:
            for feature in self.feature_dict.keys():
                X_[feature] = self.feature_dict[feature](X_)

        self.feature_list_ = X_.columns

        return X_


class TransformFeatures(BaseEstimator, TransformerMixin):
    """
    A transformer which applies specified functions to specified columns of a 
    pandas DataFrame. The constructor inputs a dict of feature name, function pairs.
    """
    def __init__(self, function_dict):
        self.function_dict = function_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        if self.function_dict.keys() is not None:
            for feature in self.function_dict.keys():
                try:
                    X_[feature] = X_[feature].apply(self.function_dict[feature])
                except KeyError:
                    print(f'TransformFeatures.transform Warning: {feature} does not exist.')

        self.feature_list_ = X_.columns

        return X_


class CustomOneHot(BaseEstimator, TransformerMixin):
    """
    A custom one-hot encoder class. The main purpose of using this class over the encoder
    provided by sklearn is that the custom encoder returns a pandas DataFrame. This way,
    we can easily track the column names.

    Drops one of the value columns, as well as the original column, 
    in order to prevent colinearity of columns.
    """
    def __init__(self, columns:list):
        self.columns = columns

    def fit(self, X:pd.DataFrame, y=None):
        return self

    def transform(self, X:pd.DataFrame, y=None):
        X_ = X.copy()
        d = {False:0, True:1}

        for column in self.columns:
            try:
                values = list(X_[column].unique())

                for value in values[1:]:
                    X_[f'{column}_is_{value}'] = X_[column] == value
                    X_[f'{column}_is_{value}'].replace(d, inplace=True)
                
                X_.drop(columns=[column], inplace=True)
            except KeyError:
                print(f'CustomOneHot.transform Warning: {column} does not exist.')

        self.feature_list_ = X_.columns

        return X_


class CustomOrdinal(BaseEstimator, TransformerMixin):
    """
    A custom one-hot encoder class. The main purpose of using this class over the encoder
    provided by sklearn is that the custom encoder returns a pandas DataFrame. This way,
    we can easily track the column names.

    Drops the supplied columns after encoding.

    The constructor takes as input a list of tuples `(column, ordering)`
    where `column` is the name of the column and `ordering` is the tuple
    of the values in their specified order.
    """

    def __init__(self, columns_orderings:tuple):
        self.columns_orderings= columns_orderings

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()

        for pair in self.columns_orderings:
            column = pair[0]
            ordering = pair[1]
            d = {}

            for i in range(len(ordering)):
                value = ordering[i]
                d[value] = i
            
            try:
                X_[column].replace(d, inplace=True)
            except KeyError:
                print(f'CustomOrdinal.transform Warning: {column} does not exist.')

        self.feature_list_ = X_.columns

        return X_


class CustomScaler(BaseEstimator, TransformerMixin):
    """
    A transformer similar to `StandardScaler` from sklearn, except it will return
    a pandas DataFrame instead. 
    """

    def __init__(self):
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()

        for column in X_.columns:
            mean = X_[column].mean()
            std  = X_[column].std()

            X_[column] = (X_[column] - mean) / std

        self.feature_list_ = X_.columns
        return X_


# TODO: Debug or rewrite this.
class RFCImputer(BaseEstimator, TransformerMixin):
    """
    A custom random forest classifier imputer. Load the constructor with an imputer model. Fit will train the imputer model.
    Transform will predict the missing values and fill them in. 
    """

    def __init__(self, features:list, missing_value=np.nan, grid_search=True, cv=5, scoring='accuracy'):
        self.features = features
        self.missing_value = missing_value
        self.grid_search = grid_search
        self.cv = 5
        self.scoring = scoring

    def fit(self, X:pd.DataFrame, y=None):
        X_ = X.copy()

        X_missing = X_[X_.values == self.missing_value]
        X_not_missing = chelp.get_complement(sup=X_, sub=X_missing)
        y_not_missing_dict = {feature : X_not_missing[feature] for feature in self.features}
        X_not_missing.drop(columns=self.features, inplace=True)

        self.feature_model_dict = {}

        param_grid = {'n_estimators':range(1, 50),
                      'max_depth':range(1, 20),
                      'max_features':range(1, 10)}

        for feature in self.features:
            y_not_missing = y_not_missing_dict[feature]
            self.feature_model_dict[feature] = RandomForestClassifier(random_state=808)
            
            if self.grid_search:
                forest_grid = RandomizedSearchCV(self.feature_model_dict[feature], 
                                                 param_distributions=param_grid, 
                                                 cv = self.cv, 
                                                 scoring=self.scoring)
                forest_grid.fit(X_not_missing, y_not_missing)
                best = forest_grid.best_params_
                n_estimators = best['n_estimators']
                max_features = best['max_features']
                max_depth    = best['max_depth']
                self.feature_model_dict[feature].fit(X=X_not_missing, y=y_not_missing, n_estimators=n_estimators, max_features=max_features, max_depth=max_depth)
            else:
                self.feature_model_dict[feature].fit(X=X_not_missing, y=y_not_missing)

        return self


    def transform(self, X:pd.DataFrame, y=None):
        X_ = X.copy()

        for feature in self.features:
            X_missing_feature = X_.loc[X_[feature] == self.missing_value].drop(columns=self.features)
            y_hat_feature = self.feature_model_dict[feature].predict(X_missing_feature)

            X_.loc[X_.feature == self.missing_value] = y_hat_feature

        return X_

        
        

#class EncodeFeatures(BaseEstimator, TransformerMixin):
    """
    A transformer which is a pipeline of encoders. The main difference from the standard encoder is 
    that we check for KeyErrors on each feature, as well as checking for duplicates. This way, 
    we can easily chain with DropFeatures in a pipeline without having to reconstruct encoders.
    """
#    def __init__(self):


    # TODO: Make custom EnodeFeatures transformer.