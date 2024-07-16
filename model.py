import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as m
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

data = pd.read_csv("datasets/train2.csv")
# model = LinearRegression
X = data[["Item_Weight", "Item_Visibility", "Item_MRP", "Outlet_Establishment_Year"]]
y = data["Item_Outlet_Sales"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)
model = LinearRegression()
model.fit(X_train, y_train)
modelPrediction = model.predict(X_test)
modelMeanAbsErr = m.mean_absolute_error(y_test, modelPrediction)
modelMeanSqrtErr = m.mean_squared_error(y_test, modelPrediction)
modelR2Score = m.r2_score(y_test, modelPrediction)

print(
    "Model Mean Absolute Error:",
    modelMeanAbsErr,
    "Model Mean Square Error:",
    modelMeanSqrtErr,
    "Rsquared score:",
    modelR2Score,
    'Training score  : {}'.format(modelPrediction.score(X_train, y_train))),
    'Test score      : {}'.format(modelPrediction.score(X_test, y_test))
)

print(modelPrediction)
