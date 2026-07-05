# Implementation for Data Cleaning

# Step 1: Import Libraries and Load Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('../documents/Titanic-Dataset.csv')
# df.info()
# df.head()

print("-----------------------------------------------------------------")

'''
Step 2: Check for Duplicate Rows

df.duplicated(): Returns a boolean Series indicating duplicate rows.

'''
# print(df.duplicated())

print("--------------------------------------------------------------")

'''
Step 3: Identify Column Data Types

List comprehension with .dtype attribute to separate categorical and numerical columns.
object dtype: Generally used for text or categorical data.
'''

cat_col = [col for col in df.columns if df[col].dtype == 'object']
num_col = [col for col in df.columns if df[col].dtype != 'object']

print('Categorical columns:', cat_col)
print('Numerical columns:', num_col)

print("--------------------------------------------------------------")

'''
Step 4: Count Unique Values in the Categorical Columns

df[cat_col].nunique(): Returns count of unique values per column.
'''

print(df[cat_col].nunique())

print("--------------------------------------------------------------")

'''
Step 5: Calculate Missing Values as Percentage

df.isnull(): Detects missing values, returning boolean DataFrame.
Sum missing across columns, normalize by total rows and multiply by 100.
'''
print(round((df.isnull().sum() / df.shape[0]) * 100, 2))
print("------------------------------------------------------------------")

'''
Step 6: Drop Irrelevant or Data-Heavy Missing Columns

df.drop(columns=[]): Drops specified columns from the DataFrame.
df.dropna(subset=[]): Removes rows where specified columns have missing values.
fillna(): Fills missing values with specified value (e.g., mean).
'''

df1 = df.drop(columns=["Name", "Ticket", "Cabin"])
print(df1)
print("------------")
print(df1.dropna(subset=["Embarked"], inplace=True))
print("------------")
df1["Age"] = df1["Age"].fillna(df1["Age"].mean())
print(df1["Age"])
print("------------------------------------------------------")

'''
Step 7: Detect Outliers with Box Plot

matplotlib.pyplot.boxplot(): Displays distribution of data, highlighting median, quartiles and outliers.
plt.show(): Renders the plot.
'''

plt.boxplot(df1["Age"], orientation='horizontal')
plt.ylabel("Variable")
plt.xlabel("Age")
plt.title("Box Plot")
plt.show()

print("--------------------------------------")

'''
Step 8: Calculate Outlier Boundaries and Remove Them

Calculate mean and standard deviation (std) using df['Age'].mean() and df['Age'].std().
Define bounds as mean ± 2 * std for outlier detection.
Filter DataFrame rows within bounds using Boolean indexing.
'''

mean = df1['Age'].mean()
print("mean value =>", mean)
std = df1['Age'].std()
print("std value =>", std)


lower_bound = mean - 2 * std
upper_bound = mean + 2 * std
print("lower_bound =>", lower_bound)
print("upper_bound =>", upper_bound)

df2 = df1[(df1['Age'] >= lower_bound) & (df1['Age'] <= upper_bound)]
print(df2)
print("--------------------------------------")


'''
Step 9: Impute Missing Data Again if Any

fillna() applied again on filtered data to handle any remaining missing values.
'''

df3 = df2.fillna(df2['Age'].mean())
print(df3.isnull().sum())

print("--------------------------------------")


'''
Step 10: Recalculate Outlier Bounds and Remove Outliers from the Updated Data

mean = df3['Age'].mean(): Calculates the average (mean) value of the Age column in the DataFrame df3.

std = df3['Age'].std(): Computes the standard deviation (spread or variability) of the Age column in df3.

lower_bound = mean - 2 * std: Defines the lower limit for acceptable Age values, set as two standard deviations below the mean.

upper_bound = mean + 2 * std: Defines the upper limit for acceptable Age values, set as two standard deviations above the mean.

df4 = df3[(df3['Age'] >= lower_bound) & (df3['Age'] <= upper_bound)]: Creates a new DataFrame df4 by selecting only rows where the Age value falls between the lower and upper bounds, effectively removing outlier ages outside this range.
'''

mean = df3['Age'].mean()
std = df3['Age'].std()

lower_bound = mean - 2 * std
upper_bound = mean + 2 * std

print('Lower Bound :', lower_bound)
print('Upper Bound :', upper_bound)

df4 = df3[(df3['Age'] >= lower_bound) & (df3['Age'] <= upper_bound)]

print(df4)

print("------------------------------------------------")

'''
Step 11: Data validation and verification

Data validation and verification involve ensuring that the data is accurate and consistent by comparing it with external sources or expert knowledge. 

For the machine learning prediction we separate independent and target features

Here we will consider 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', and 'Embarked' as independent features.

Survived as target variables because PassengerId will not affect the survival rate
'''

X = df3[['Pclass','Sex','Age', 'SibSp','Parch','Fare','Embarked']]
Y = df3['Survived']
print(X, Y)
print("-----------------------------------------------------------------")

'''
Step 12: Data formatting

1. Min-Max Scaling - Scales values between 0 and 1.

2. Standardization (Z-score scaling) - Transforms data so that it has a mean of 0 and a standard deviation of 1.

Z = (X - μ) / σ

X = Data
μ = Mean value of X
σ = Standard deviation of X
'''

scaler = MinMaxScaler(feature_range=(0, 1))
print(scaler)

# Safe and explicit way to grab only numeric columns
num_col_ = X.select_dtypes(include=['number']).columns.tolist()
print(num_col_)

# Avoid modifying your original DataFrame by explicitly creating a copy
x1 = X.copy()
x1[num_col_] = scaler.fit_transform(x1[num_col_])
print(x1.head())