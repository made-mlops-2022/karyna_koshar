import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class CategoricalTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, cols=None):
        if isinstance(cols, str):
            self.cols = [cols]
        else:
            self.cols = cols

    def fit(self, X, y=None):
        if self.cols is None:
            self.cols = X.columns

        for col in self.cols:
            if col not in X:
                raise ValueError('Column ' + col + ' not in X')

        self.maps = dict() 
        for col in self.cols:
            tmap = dict()
            uniques = X[col].unique()
            col_sum = X[col].count()
            for unique in uniques:
                tmap[unique] = X[X[col] == unique][col].count() / col_sum
            self.maps[col] = tmap
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ['maps'])
        Xo = X.copy()
        for col, tmap in self.maps.items():
            vals = np.full(X.shape[0], np.nan)
            for val, mean_target in tmap.items():
                vals[X[col] == val] = mean_target
            Xo[col] = vals
        return Xo

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
