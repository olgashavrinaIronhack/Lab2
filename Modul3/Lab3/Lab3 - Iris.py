# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 14:53:08 2022

@author: oshav
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
iris=pd.read_csv(r'C:/Users/oshav/DAFT_NOV_21_01/module_3/3. Data-Cleaning-Challenge/iris-data.csv')
iris.shape
iris.describe()
iris.isna().sum()
iris["class"].unique()


classes={'Iris-setosa':'setosa','Iris-setossa':'setosa','Iris-versicolor':'versicolor','versicolor':'versicolor','Iris-virginica':'virginica'}
iris["class"]=iris['class'].map(classes)
plt.boxplot(iris)
iris
iris.boxplot()
iris.loc[iris['petal_width_cm']].isna()
iris['petal_width_cm'].fillna(round(iris['petal_width_cm'][iris['class'] == 'setosa'].mean(),2), inplace=True)
iris.isna().sum()
iris
sns.heatmap(data=iris,)
corr=iris.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
sns.scatterplot(data=iris, x='sepal_length_cm',y='sepal_width_cm')
sns.scatterplot(data=iris, x='sepal_length_cm',y='petal_width_cm')
iris_1=iris[iris['petal_width_cm']==0]
sns.histplot(data=iris,x='class',hue='petal_width_cm')
