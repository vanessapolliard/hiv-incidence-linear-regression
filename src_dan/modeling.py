import copy
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn import datasets, linear_model
from utils import XyScaler
import os
 

class Modeling(object):

    def __init__(self, df, y_index, x_index, model, kfolds, cleaner=None):
        self.df = df
        self.y_index = y_index
        self.x_index = x_index
        self.model = model
        self.kfolds = kfolds
        self.cleaner = cleaner
    
    def df_to_x_y(self):
        self.X = df.iloc[:, self.x_index:].to_numpy()
        self.y = df.iloc[:, self.y_index].to_numpy()

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y)


    def standardize(self):
        standardizer = XyScaler()
        standardizer.fit(self.X_train, self.y_train)
        self.X_train_standardized, self.y_train_standardized = standardizer.transform(self.X_train, self.y_train)
        self.X_test_standardized, self.y_test_standardized = standardizer.transform(self.X_test, self.y_test)

    def fit(self):
        self.model.fit(self.X_train_standardized, self.y_train_standardized)
         

    def predict(self):
        y_pred = self.model.predict(self.X_test_standardized)
        error = self.rmse(self.y_test_standardized, y_pred)
        return error
    
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
            # X_train = cleaner.clean(self.X[train])
            # X_test = cleaner.clean(self.X[test])

            kf_model.fit(self.X_train_standardized[train], self.y_train_standardized[train])
            y_pred = kf_model.predict(self.X_test_standardized[test])
            error[index] = self.rmse(self.y_test_standardized[test], y_pred)
            index += 1
        
        return np.mean(error)
        


if __name__ == '__main__':
    DATA_DIRECTORY = os.path.join(os.path.split(os.getcwd())[0], 'data')
    df = pd.read_csv(os.path.join(DATA_DIRECTORY, 'main-df.csv'))
    df = df.dropna()
    df = df[df['HIVincidence']<750]

    modeler = Modeling(df, 6, 8, linear_model.LinearRegression(), 10)
    modeler.df_to_x_y()
    modeler.split_data()
    modeler.standardize()
    modeler.fit()
    error = modeler.predict()
    

    # feature_names = data1['feature_names']
    # raw_data_x = data[0][:100]
    # raw_data_y = data[1][:100]