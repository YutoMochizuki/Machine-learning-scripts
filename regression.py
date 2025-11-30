'''
regression script (linear, ransac, ridge, lasso, elasticnet, svs, random forest)
'''

import datetime
import glob
import io
import math
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm
from matplotlib import colors
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import constants as const

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor


plt.rcParams['font.size'] = 10
plt.rcParams["xtick.top"] = True
plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["ytick.right"] = True
plt.rcParams['font.family'] ='Times New Roman'
plt.rcParams['xtick.direction'] = "in"
plt.rcParams['ytick.direction'] = "in"
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams["legend.loc"] = "best"
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.framealpha"] = 1.0
plt.rcParams["legend.facecolor"] = "white"
plt.rcParams["legend.edgecolor"] = "black"
plt.rcParams["legend.fancybox"] = False
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["axes.grid"] = False

sns.set_style("ticks", {
    'axes.facecolor': "1.0",
    'xtick.major.size': 10.0,
    'ytick.major.size': 10.0,
    'xtick.minor.size': 6.0,
    'ytick.minor.size': 6.0,
    'xtick.direction': u'in',
    'ytick.direction': u'in',
    })
sns.set_context("talk", 1.0)
font = {"family":"Noto Sans CJK JP"}
mpl.rc('font', **font)
palette = sns.color_palette()


random_num = 0


def pickl_to_pd(table_list):
    '''
    pandas pikle
    '''
    df = pd.concat([pd.read_pickle(x) for x in table_list])
    return df


def txt_to_pandas(file_path, columns):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            #  1行目をスキップ
            if line.startswith('#'):
                continue
            values = line.strip().split()  # スペースで分割
            data.append(values)
    # データフレームを作成
    df = pd.DataFrame(data, columns=columns)
    return df


def preprocess(df):
    # 欠損値処理
    if df.isnull().values.any():
        # 行を削除
        df = df.dropna(how='any')

        # 平均値で補完
        # imputer = SimpleImputer(strategy='mean')
        # df = imputer.fit_transform(df)

        df = df.reset_index(drop=True)

    df = pd.get_dummies(df, dtype=int)
    df = df.astype(float)
    return df


def training_normalization(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_num)

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # 平均、分散
    print('mean: ', sc.mean_)
    print('var: ', sc.var_)

    return X_train_std, X_test_std, y_train, y_test


def linear_regression(X_train_std, X_test_std, y_train, y_test,explained_variance, objected_variance, objected_variance_index):
    slr = LinearRegression()
    slr.fit(X_train_std, y_train)
    y_train_pred = slr.predict(X_train_std)
    y_test_pred = slr.predict(X_test_std)

    print('回帰係数：', slr.coef_[0])
    print('切片：', slr.intercept_)
    print('決定係数：', slr.score(X_test_std, y_test))

    plot(X_test_std, y_test_pred, y_test_pred, explained_variance, objected_variance)

    return y_train_pred, y_test_pred


def ransac_regression(X_train_std, X_test_std, y_train, y_test, explained_variance, objected_variance, objected_variance_index):
    ransac = RANSACRegressor(LinearRegression(), max_trials=100, min_samples=50, residual_threshold=5.0, random_state=random_num, loss='absolute_error')
    ransac.fit(X_train_std, y_train)
    y_train_pred = ransac.predict(X_train_std)
    y_test_pred = ransac.predict(X_test_std)

     # 正常値をtrue
    inlier_mask = ransac.inlier_mask_
    # 外れ値をtrue
    outlier_mask = np.logical_not(inlier_mask)

    print('回帰係数：', ransac.estimator_.coef_[0])
    print('切片：', ransac.estimator_.intercept_)
    print('決定係数：', ransac.score(X_test_std, y_test))

    plot(X_test_std, y_test, y_test_pred, explained_variance, objected_variance)

    return y_train_pred, y_test_pred


def ridge_regression(X_train_std, X_test_std, y_train, y_test, explained_variance, objected_variance, objected_variance_index):
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_std, y_train)
    y_train_pred = ridge.predict(X_train_std)
    y_test_pred = ridge.predict(X_test_std)

    print('回帰係数：', ridge.coef_[0])
    print('切片：', ridge.intercept_)
    print('決定係数：', ridge.score(X_test_std, y_test))

    plot(X_test_std, y_test, y_test_pred, explained_variance, objected_variance)
    return y_train_pred, y_test_pred


def lasso_regression(X_train_std, X_test_std, y_train, y_test, explained_variance, objected_variance, objected_variance_index):
    lasso = Lasso(alpha=1.0)
    lasso.fit(X_train_std, y_train)
    y_train_pred = lasso.predict(X_train_std)
    y_test_pred = lasso.predict(X_test_std)

    print('回帰係数：', lasso.coef_[0])
    print('切片：', lasso.intercept_)
    print('決定係数：', lasso.score(X_test_std, y_test))

    plot(X_test_std, y_test, y_test_pred, explained_variance, objected_variance)
    return y_train_pred, y_test_pred


def elasticnet_regression(X_train_std, X_test_std, y_train, y_test, explained_variance, objected_variance, objected_variance_index):
    elasticnet = ElasticNet(alpha=1.0, l1_ratio=0.5)
    elasticnet.fit(X_train_std, y_train)
    y_train_pred = elasticnet.predict(X_train_std)
    y_test_pred = elasticnet.predict(X_test_std)

    print('回帰係数：', elasticnet.coef_[0])
    print('切片：', elasticnet.intercept_)
    print('決定係数：', elasticnet.score(X_test_std, y_test))

    plot(X_test_std, y_test, y_test_pred, explained_variance, objected_variance)
    return y_train_pred, y_test_pred


def svr_regression(X_train_std, X_test_std, y_train, y_test, explained_variance, objected_variance, objected_variance_index):
    svr = SVR(kernel='linear')
    svr.fit(X_train_std, y_train)
    y_train_pred = svr.predict(X_train_std)
    y_test_pred = svr.predict(X_test_std)

    print('回帰係数：', svr.coef_[0])
    print('切片：', svr.intercept_)
    print('決定係数：', svr.score(X_test_std, y_test))

    plot(X_test_std, y_test, y_test_pred, explained_variance, objected_variance)
    return y_train_pred, y_test_pred


def random_forest(X_train_std, X_test_std, y_train, y_test, explained_variance, objected_variance, objected_variance_index):
    rf = RandomForestRegressor(n_estimators=100, random_state=random_num, n_jobs=2)
    rf.fit(X_train_std, y_train)
    y_train_pred = rf.predict(X_train_std)
    y_test_pred = rf.predict(X_test_std)

    # print('回帰係数：', rf.feature_importances_)
    # print('切片：', rf.estimator_.intercept_)
    # print('決定係数：', rf.score(X, y))

    plot(X_test_std, y_test, y_test_pred, explained_variance, objected_variance)
    return y_train_pred, y_test_pred


def cost_func(X, y, n_cluster):
    cost = 0
    for k in range(int(n_cluster)):
        cost += sum((X[y==k,0] - X[y==k,0].mean(axis=0))**2 + (X[y==k,1] - X[y==k,1].mean(axis=0))**2)
    return cost


def plot(X, y, y_pred, explained_variance, objected_variance):
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color='blue', label='Data')
    plt.plot(X, y_pred, color='red', label='Regression Line')
    # plt.scatter(X[:, objected_variance_index], y, color='blue', label='Data')
    # plt.plot(X[:, objected_variance_index], y_pred, color='red', label='RANSAC Regression Line')
    plt.xlabel(explained_variance)
    plt.ylabel(objected_variance)
    plt.legend()
    plt.tight_layout()
    plt.savefig('regression.png')
    plt.close()


def plot_residuals(y_train, y_pred_train, y_test, y_pred_test):
    plt.figure(figsize=(10, 8))
    plt.scatter(y_train, y_train - y_pred_train, c='blue', marker='o', label='Train data')
    plt.scatter(y_test, y_test - y_pred_test, c='lightgreen', marker='s', label='Test data')
    plt.xlabel('Observed values')
    plt.ylabel('Residuals')
    plt.hlines(y=0, xmin=min(y_train.min(), y_test.min()), xmax=max(y_train.max(), y_test.max()), color='red')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('residuals.png')
    plt.close()


def main():
    # sepは改行
    df = pd.read_csv('../doc/housing.csv', header=None)
    df.columns = ['INDEX', 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
    df = df[cols]
    df = preprocess(df)
    print(df.head())

    explained_variance = 'RM'
    objected_variance = 'MEDV'
    objected_variance_index = cols.index(objected_variance) - 1
    X = df[[explained_variance]].values
    # X = df.iloc[:, :-1].values
    y = df[objected_variance].values

    X_train_std, X_test_std, y_train, y_test = training_normalization(X, y)

    # y_pred_train, y_pred_test = linear_regression(X_train_std, X_test_std, y_train, y_test, explained_variance, objected_variance, objected_variance_index)

    # y_pred_train, y_pred_test = ransac_regression(X_train_std,  X_test_std, y_train, y_test, explained_variance, objected_variance, objected_variance_index)

    # y_pred_train, y_pred_test = ridge_regression(X_train_std, X_test_std, y_train, y_test, explained_variance, objected_variance, objected_variance_index)

    y_pred_train, y_pred_test = lasso_regression(X_train_std, X_test_std, y_train, y_test, explained_variance, objected_variance, objected_variance_index)

    # y_pred_train, y_pred_test = elasticnet_regression(X_train_std, X_test_std, y_train, y_test, explained_variance, objected_variance, objected_variance_index)

    # y_pred_train, y_pred_test = svr_regression(X_train_std, X_test_std, y_train, y_test, explained_variance, objected_variance, objected_variance_index)

    # y_pred_train, y_pred_test = random_forest(X_train_std, X_test_std, y_train, y_test, explained_variance, objected_variance, objected_variance_index)

    plot_residuals(y_train, y_pred_train, y_test, y_pred_test)


if __name__ == "__main__":
    main()
