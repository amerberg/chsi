from sklearn.base import BaseEstimator

class Truncator(BaseEstimator):
    def __init__(self, minimum, maximum):
        self.minimum = minimum
        self.maximum = maximum
        
    def fit(self, X, Y):
        return self
        
    def transform(self, X):
        X[(X < self.minimum) | (X > self.maximum)] = 0
        return X
        
        