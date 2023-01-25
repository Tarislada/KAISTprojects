import xgboost as xgb
from xgboost import XGBClassifier
import pandas as pd
import sklearn
import numpy as np
import copy as cp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from typing import Tuple
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Read data
a = pd.read_csv('C:/Users/endyd/OneDrive/Onedrive-CK/OneDrive/Gradschool/Kaist/convertest.csv',header=None)
data = a.loc[:,0:419]
label = a.loc[:,420]
data_dmatrix = xgb.DMatrix(data=data,label=label)

# Set model
params = {"objective":"multi:softmax",'colsample_bytree':0.3,'learning_rate':0.1,'max_depth':5,'alpha':10,'num_class':5}
xgb_cv = xgb.cv(dtrain=data_dmatrix,params=params,nfold=3,num_boost_round=50,early_stopping_rounds=10,metrics="auc",as_pandas=True,seed=123)
xgb_class = xgb.XGBClassifier(**params)
xgb_cv.head()

# Model training
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data,label,test_size=0.3, random_state=0)
#xgb_class.fit(X_train, y_train)
#y_pred = xgb_class.predict(X_test)

# Cross validation
def cross_val_predict(model, kfold: KFold, X: np.array, y: np.array) -> Tuple[np.array, np.array, np.array]:
    model_ = cp.deepcopy(model)

    no_classes = len(np.unique(y))

    actual_classes = np.empty([0], dtype=int)
    predicted_classes = np.empty([0], dtype=int)
    predicted_proba = np.empty([0, no_classes])

    for train_ndx, test_ndx in kfold.split(X):

        train_X, train_y, test_X, test_y = X[train_ndx], y[train_ndx], X[test_ndx], y[test_ndx]

        actual_classes = np.append(actual_classes, test_y)

        model_.fit(train_X, train_y)
        predicted_classes = np.append(predicted_classes, model_.predict(test_X))

        try:
            predicted_proba = np.append(predicted_proba, model_.predict_proba(test_X), axis=0)
        except:
            predicted_proba = np.append(predicted_proba, np.zeros((len(test_X), no_classes), dtype=float), axis=0)

    return actual_classes, predicted_classes, predicted_proba

sklearn.metrics.confusion_matrix()
