from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso

def prediction(model,split,model_name):

    X_train, X_test, y_train, y_test = split
    pred = model.predict(X_test)
    print("Model:",model_name)
    print("MAE:", mean_absolute_error(y_test, pred),"RMSE:",math.sqrt(mean_squared_error(y_test, pred)))
    # print("MAE:",mean_absolute_error(y_train, model.predict(X_train)),"RMSE:",math.sqrt(mean_squared_error(y_train, model.predict(X_train))))
    return pred

# Create an object called iris with the iris data
diabetes = load_diabetes()

X = diabetes.data
Y = diabetes.target

# X = X[:,4:]
# X = X.reshape(-1,1)
# plt.scatter(X[:,1],Y)
# plt.show()


#Splitting data
split = model_selection.train_test_split(X, Y, test_size=0.33)

X_train, X_test, y_train, y_test = split



ols = linear_model.LinearRegression().fit(X_train,y_train)
pred_ols = prediction(ols,split,"Linear regression")



GBR = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls').fit(X_train, y_train)
pred_GBR = prediction(GBR,split,"Gradient Boosting Regressor")

RFR = RandomForestRegressor(random_state=0, n_estimators=100,max_depth=6).fit(X_train, y_train)
pred_RFR = prediction(RFR,split,"Random Forest Regressor")


KNR = KNeighborsRegressor(n_neighbors=2).fit(X_train, y_train)
pred_KNR = prediction(KNR,split,"KNeighborsRegressor")

SVR = SVR(kernel='linear').fit(X_train,y_train)
pred_SVR = prediction(KNR,split,"Epsilon-Support Vector Regression")

lasso = Lasso(random_state=0).fit(X_train,y_train)
pred_SVR = prediction(lasso,split,"Lasso")


# print(cross_val_score(ols, X, Y).mean(),cross_val_score(GBR, X, Y).mean(),cross_val_score(RFR, X, Y).mean())

plt.scatter(pred_ols,y_test)
plt.scatter(pred_GBR,y_test,c='r')
plt.scatter(pred_RFR,y_test,c='g')
# plt.show()
