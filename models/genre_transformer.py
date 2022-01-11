import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Build a custom transformer which will extract the starting verb of a sentence
class GenreTransformer(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class
    
    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Create dummy variables from the data and put the results into a dataframe called dummies
        dummies = pd.get_dummies(X['genre'])
        X_transformed = dummies

        return X_transformed