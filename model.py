import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as m
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    cross_val_score,
)
from sklearn.ensemble import RandomForestRegressor
import pickle

data = pd.read_csv("datasets/train2.csv")


## Model
# model = LinearRegression
X = data.drop(columns=["Item_Outlet_Sales", "Item_Identifier"], axis=1)
y = data["Item_Outlet_Sales"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)
model = RandomForestRegressor(n_estimators=50, min_samples_leaf=4, bootstrap=True)
model.fit(X_train, y_train)
predictionModel = model.predict(X_test)

## Hypertuning parametres for better model perferomance
parameters = {
    "bootstrap": [True, False],
    "n_estimators": [25, 50, 100, 150],
    "max_features": ["sqrt", "log2", None],
    "min_samples_leaf": [1, 2, 4],
}

randomSearch = RandomizedSearchCV(
    estimator=model,
    param_distributions=parameters,
    n_iter=30,
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1,
)

randomSearch.fit(X, y)
print(randomSearch.best_params_)
print(randomSearch.best_score_)

## Model evaluation
modelMeanAbsErr = m.mean_absolute_error(y_test, predictionModel)
modelMeanSqrtErr = m.mean_squared_error(y_test, predictionModel)
modelR2Score = m.r2_score(y_test, predictionModel)
cv_scores = cross_val_score(model, X, y, cv=3, scoring="neg_mean_squared_error")
rmse_scores = np.sqrt(-cv_scores)

print(
    "Model Mean Absolute Error:",
    modelMeanAbsErr,
    "Model Mean Square Error:",
    modelMeanSqrtErr,
    "Rsquared score:",
    modelR2Score,
    "Training score:",
    (model.score(X_train, y_train)),
    "Test score:",
    (model.score(X_test, y_test)),
    "Cross-validated RMSE scores:",
    rmse_scores,
)


## Saving trained model
with open("notinoAssignmentSalesPrediction.pkl", "wb") as file:
    pickle.dump(predictionModel, file)
