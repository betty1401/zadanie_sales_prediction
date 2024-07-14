import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import LabelEncoder

# Importing datasets
test = pd.read_csv("datasets/test.csv")
train = pd.read_csv("datasets/train.csv")

print(test.head())
print(test.columns)

for columns in test:
    print(train[columns].value_counts())

# Exploring the data in general by looking at its structure
info = train.describe(include="all")
# null_values = train.isnull().sum().sort_values(ascending=False)
# print(null_values)

## Looking at categories through bar charts
categoricalCols = [
    "Item_Fat_Content",
    "Item_Type",
    "Outlet_Identifier",
    "Outlet_Size",
    "Outlet_Location_Type",
    "Outlet_Type",
]
categoriesTempDf = train[categoricalCols]
catList = categoriesTempDf.columns.tolist()


for catCols in categoriesTempDf:
    plt.figure()
    catCount = categoriesTempDf[catCols].value_counts()
    catCount.plot(kind="bar")
    plt.xlabel(catCols)
    plt.ylabel("Count")
    # plt.show()

## Preprocessing and data preparation
train["Item_Fat_Content"] = train["Item_Fat_Content"].replace(
    {
        "Low Fat": "Low Fat",
        "LF": "Low Fat",
        "low fat": "Low Fat",
        "Regular": "Regular",
        "reg": "Regular",
    }
)

train["Outlet_Size"] = train["Outlet_Size"].fillna(train["Outlet_Size"].mode()[0])
train["Item_Weight"] = train["Item_Weight"].fillna(train["Item_Weight"].mean())


for catCols in catList:
    train[catCols] = LabelEncoder().fit_transform(train[catCols])
