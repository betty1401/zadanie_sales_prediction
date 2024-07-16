import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Importing datasets
test = pd.read_csv("datasets/test.csv")
train = pd.read_csv("datasets/train.csv")


print(train.head())
print(train.columns)
print(train.dtypes)

for columns in test:
    print(train[columns].value_counts())

# Exploring the data in general by looking at its structure
info = train.describe(include="all")
null_values = train.isnull().sum().sort_values(ascending=False)
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
    # plt.figure()
    catCount = categoriesTempDf[catCols].value_counts()
    catCount.plot(kind="bar")
    plt.xlabel(catCols)
    plt.ylabel("Count")
    # plt.show()

## Preprocessing and data preparation
# Standardization of category names
train["Item_Fat_Content"] = train["Item_Fat_Content"].replace(
    {
        "Low Fat": "Low Fat",
        "LF": "Low Fat",
        "low fat": "Low Fat",
        "Regular": "Regular",
        "reg": "Regular",
    }
)

# Removing null values
train["Outlet_Size"] = train["Outlet_Size"].fillna(train["Outlet_Size"].mode()[0])
train["Item_Weight"] = train["Item_Weight"].fillna(train["Item_Weight"].mean())

# Transforming categorical data to numerical values
for catCols in catList:
    train[catCols] = LabelEncoder().fit_transform(train[catCols])

# Finding correlation between variables in order to determine right features for model
correlation = train.corr()
print(correlation)
sns.heatmap(correlation, annot=True)
plt.show()

# Identifying and removing outliers
numCols = ["Item_Weight", "Item_Visibility", "Item_MRP"]
sns.boxplot(train[numCols])
plt.show()

Q1 = train["Item_Visibility"].quantile(0.25)
Q3 = train["Item_Visibility"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

upperThreshold = np.where(train["Item_Visibility"] >= upper)[0]
lowerThreshold = np.where(train["Item_Visibility"] <= lower)[0]

# Removing the outliers
train.drop(index=upperThreshold, inplace=True)
train.drop(index=lowerThreshold, inplace=True)
sns.boxplot(train["Item_Visibility"])
plt.show()

## Saving preprocesed dataset
train.to_csv("datasets/train2.csv")
