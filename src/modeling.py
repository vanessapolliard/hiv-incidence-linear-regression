import copy
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
 

class Modeling(object):

    def __init__(self, X, y, model, kfolds, cleaner=None):
        self.X = X
        self.y = y
        self.model = model
        self.kfolds = kfolds
        self.cleaner = cleaner

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y)

    def fit(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        predicted = self.model.predict(self.X_test)
        return predicted
    
    def rmse(self, y_true, y_pred):
        mse = ((y_true - y_pred)**2).mean()
        return np.sqrt(mse)
    
    def cross_val_score(self):
        kf = KFold(n_splits=self.kfolds)
        error = np.empty(self.kfolds)
        index = 0
        kf_model = copy.deepcopy(self.model)
        for train, test in kf.split(self.X):
            # Clean features
            X_train = cleaner.clean(self.X[train])
            X_test = cleaner.clean(self.X[test])
            
            kf_model.fit(X_train, self.y[train])
            y_pred = kf_model.predict(X_test)
            error[index] = self.rmse(self.y[test], y_pred)
            index += 1
        
        return np.mean(error)
        
