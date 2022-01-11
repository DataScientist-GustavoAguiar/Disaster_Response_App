import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Build a custom transformer which will extract the starting verb of a sentence
class LengthTransformer(BaseEstimator, TransformerMixin):
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
        X_transformed = X.to_frame()
        X_transformed['word_count'] = X.apply(lambda x: len(str(x).split(" ")))
        X_transformed['char_count'] = X.apply(lambda x: sum(len(word) for word in str(x).split(" ")))
        X_transformed['sentence_count'] = X.apply(lambda x: len(str(x).split(".")))
        X_transformed['avg_word_length'] = X_transformed['char_count'] / X_transformed['word_count']
        X_transformed['avg_sentence_lenght'] = X_transformed['word_count'] / X_transformed['sentence_count']
        X_transformed = X_transformed.drop(['message'], axis=1)

        return X_transformed