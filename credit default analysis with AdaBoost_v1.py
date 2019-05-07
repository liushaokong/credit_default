#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# 1. import necessary modules
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score


# 2. data prep
data = pd.read_csv('./UCI_Credit_Card.csv')
# print(data.shape)  # (30000, 25)
# print(data.describe())  # to show all?
# next_month = data['default.payment.next.month'].value_counts()
# print(next_month)
# df = pd.DataFrame({'default.payment.next.month': next_month.index, 'values': next_month.values})
# plt.figure(figsize=(6, 6))
# plt.title('default customers\n (default: 1, un_default: 0)')
# sns.set_color_codes("pastel")
# sns.barplot(x='default.payment.next.month', y="values", data=df)
# locs, labels = plt.xticks()
# plt.show()

data.drop(['ID'], inplace=True, axis=1)
target = data['default.payment.next.month'].values

columns = data.columns.tolist()
columns.remove('default.payment.next.month')
features = data[columns].values

train_x, test_x, train_y, test_y = train_test_split(features, target, test_size=0.30, stratify=target, random_state=1)


# 3. clf
svc = SVC(random_state=1, kernel='rbf', C=1.5, gamma=0.05)

classifiers = [
    # svc,
    RandomForestClassifier(random_state=1, criterion='gini'),
    AdaBoostClassifier(random_state=1, algorithm='SAMME')]  # algorithm should be change to SAMME

classifier_names = [
    # 'svc',
    'RandomForestClassifier',
    'AdaBoostClassifier']



classifier_param_grid = [
    # {'svc__C': [1.0, 1.5], 'svc__gamma': [0.01, 0.05]},
    {'RandomForestClassifier__n_estimators': [3, 5, 6]},
    {'AdaBoostClassifier__base_estimator': [None, svc], 'AdaBoostClassifier__n_estimators': [10]}  # 10, 50, 100
]


def GridSearchCV_work(pipeline, train_x, train_y, test_x, test_y, param_grid, score='accuracy'):
    response = {}
    gridsearch = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring=score)
    search = gridsearch.fit(train_x, train_y)
    print("GridSearch最优参数：", search.best_params_)
    print("GridSearch最优分数： %0.4lf" % search.best_score_)
    predict_y = gridsearch.predict(test_x)
    print("准确率 %0.4lf" % accuracy_score(test_y, predict_y))
    response['predict_y'] = predict_y
    response['accuracy_score'] = accuracy_score(test_y, predict_y)
    return response


for model, model_name, model_param_grid in zip(classifiers, classifier_names, classifier_param_grid):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        (model_name, model)  # model_name, model
    ])
    # model_param_grid
    result = GridSearchCV_work(pipeline, train_x, train_y, test_x, test_y, model_param_grid, score='accuracy')

