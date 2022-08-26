import pandas as pd
import numpy as np
import warnings 
import math

import matplotlib.pyplot as plt

import plotly.express as px
import plotly.offline as py
import plotly.io as pio
import plotly.graph_objs as go

import seaborn as sns
import seaborn as sns

from scipy.stats import norm, skew

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import pickle

df = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv')

df_clean = df.copy()
df_clean = df_clean[(df_clean["BMI"] > 0 ) & (df_clean["BloodPressure"] > 0) & (df_clean["Glucose"] > 0)]
df_clean


X = df_clean.drop('Outcome', axis=1)
y = df_clean['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=28)

df_train = pd.concat([X_train, y_train], axis=1)

clf = DecisionTreeClassifier(criterion='entropy',random_state=0)

clf.fit(X_train, y_train)

clf_pred=clf.predict(X_test)

cm = confusion_matrix(y_test, clf_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
display_labels=clf.classes_)
disp.plot()

plt.show()

#Cross Validation

params = {'criterion': ['gini', 'entropy'], 'max_depth': range(1,10), 'min_samples_split':range(1,10), 'min_samples_leaf': range(1,5)}

grid = GridSearchCV(DecisionTreeClassifier(random_state=0), params, verbose=1, n_jobs=-1,cv=3)

grid.fit(X_train, y_train)

model_cv = grid.best_estimator_

y_pred = model_cv.predict(X_test)

cm = confusion_matrix(y_test, y_pred, labels=grid.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
display_labels=grid.classes_)
disp.plot()

plt.show()

filename = '../models/model_cv.sav'
pickle.dump(model_cv, open(filename, 'wb'))