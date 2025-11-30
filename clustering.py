'''
clustering script (ppn, logistic, kmeans)
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
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score


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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_num, stratify=y)

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    # 平均、分散
    print('mean: ', sc.mean_)
    print('var: ', sc.var_)
    return X_train_std, X_test_std, y_train, y_test


def ppn(X_train_std, X_test_std, y_train, y_test):
    # max_iter:エポック数＝学習回数
    # eta0:学習率＝重みをどのくらい更新するか

    print('----- perceptron -----')
    n_cluster = len(np.unique(y_test))
    ppn = Perceptron(max_iter=40, eta0=0.1, random_state=random_num, shuffle=True)
    ppn.fit(X_train_std, y_train)
    y_pred = ppn.predict(X_test_std)
    cost_train = cost_func(X_train_std, y_train, n_cluster)
    cost_test = cost_func(X_test_std, y_test, n_cluster)
    cost = (cost_train, cost_test)

    print('Precision: %.2f' % precision_score(y_test, y_pred, average='macro'))
    print('Recall: %.2f' % recall_score(y_test, y_pred, average='macro'))
    print('F1-score: %.2f' % f1_score(y_test, y_pred, average='macro'))

    score = cross_val_score(ppn, X_train_std, y_train, cv=5)
    print('Cross-validation scores accuracy: ', (np.mean(score), np.std(score)))

    confusion_plot(y_test, y_pred)

    return y_pred, ppn, cost


def logistic(X_train_std, X_test_std, y_train, y_test):
    # C:教師データの分類の間違いに対して、モデルが学習する識別境界線をどのくらい厳しくするのか

    # L1(ラッソ回帰）...データの"特徴量の削減"により、識別境界線の一般化を図るペナルティ
    # L2（リッジ回帰）...データ全体の"重みの減少"により、識別境界線の一般化を図るペナルティ
    # elasticnet...L1,L2の両方を使用するペナルティ
    # none...正則化を行わない

    # ovr...クラスに対して「属する/属さない」の二値分類の問題に適している
    # multinomial...各クラスに分類される確率も考慮され、「属する/属さない」だけではなく「どれくらい属する可能性があるか」を扱う問題に適している

    print('----- logistic regression -----')
    n_cluster = len(np.unique(y_test))
    lg = LogisticRegression(C=100.0, random_state=random_num, solver='lbfgs', multi_class='ovr')
    lg.fit(X_train_std, y_train)
    y_pred = lg.predict(X_test_std)
    cost_train = cost_func(X_train_std, y_train, n_cluster)
    cost_test = cost_func(X_test_std, y_test, n_cluster)
    cost = (cost_train, cost_test)

    print('Precision: %.2f' % precision_score(y_test, y_pred, average='macro'))
    print('Recall: %.2f' % recall_score(y_test, y_pred, average='macro'))
    print('F1-score: %.2f' % f1_score(y_test, y_pred, average='macro'))

    score = cross_val_score(lg, X_train_std, y_train, cv=5)
    print('Cross-validation scores accuracy: ', (np.mean(score), np.std(score)))

    confusion_plot(y_test, y_pred)
    return y_pred, lg, cost


def kmeans(X_train_std, X_test_std, y_train, y_test):
    print('----- k-means clustering -----')
    n_cluster = len(np.unique(y_test))
    kmeans = KMeans(n_clusters=n_cluster, random_state=random_num)
    kmeans.fit(X_train_std)
    y_pred = kmeans.predict(X_test_std)
    cost_train = cost_func(X_train_std, y_pred, n_cluster)
    cost_test = cost_func(X_test_std, y_pred, n_cluster)
    cost = (cost_train, cost_test)

    print('Precision: %.2f' % precision_score(y_test, y_pred, average='macro'))
    print('Recall: %.2f' % recall_score(y_test, y_pred, average='macro'))
    print('F1-score: %.2f' % f1_score(y_test, y_pred, average='macro'))

    score = cross_val_score(kmeans, X_train_std, y_train, cv=5)
    print('Cross-validation scores accuracy: ', (np.mean(score), np.std(score)))

    confusion_plot(y_test, y_pred)
    return y_pred, kmeans, cost


def decision_tree(X_train_std, X_test_std, y_train, y_test):
    print('----- decision tree -----')
    n_cluster = len(np.unique(y_test))
    dt = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=random_num)
    dt.fit(X_train_std, y_train)
    y_pred = dt.predict(X_test_std)
    cost_train = cost_func(X_train_std, y_train, n_cluster)
    cost_test = cost_func(X_test_std, y_test, n_cluster)
    cost = (cost_train, cost_test)

    return y_pred, dt, cost


def random_forest(X_train_std, X_test_std, y_train, y_test):
    print('----- random forest -----')
    n_cluster = len(np.unique(y_test))
    rf = RandomForestClassifier(criterion='gini', n_estimators=100, random_state=random_num, n_jobs=2)
    rf.fit(X_train_std, y_train)
    y_pred = rf.predict(X_test_std)
    cost_train = cost_func(X_train_std, y_train, n_cluster)
    cost_test = cost_func(X_test_std, y_test, n_cluster)
    cost = (cost_train, cost_test)
    return y_pred, rf, cost


def cost_func(X, y, n_cluster):
    cost = 0
    for k in range(int(n_cluster)):
        cost += sum((X[y==k,0] - X[y==k,0].mean(axis=0))**2 + (X[y==k,1] - X[y==k,1].mean(axis=0))**2)
    return cost


def confusion_plot(y_test, y_pred):
    # 混同行列の生成
    cm = confusion_matrix(y_test, y_pred)

    # Confusion matrixの表示
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)

    # 描画された数値テキストを非表示にする。
    # 下記コードを書いておかないとConfusionMatrixDisplay本来の数値が表示されてしまう。
    for text in disp.ax_.texts:
        text.set_visible(False)

    # 各区画に要素数とパーセンテージを表示（行ごとの合計に対する割合）
    for (i, j), val in np.ndenumerate(cm): # 混同行列の各要素をインデックス（i, j）と値（val）として列挙
        row_sum = np.sum(cm[i, :])  # 混同行列の行iの合計を計算
        percentage = val / row_sum * 100 # 要素数を行の合計で割ってパーセンテージを計算
        color = 'white' if cm[i, j] > np.max(cm) / 2 else 'black' # 区画の値が最大値の半分を超える場合は白、そうでない場合は黒で文字を表示
        plt.text(j, i, f'{val}\n({percentage:.2f}%)', ha='center', va='center', color=color, fontsize=12) # 各区画に要素数とパーセンテージを表示。テキストを中央に配置し、フォントサイズは12に設定。

    # X軸ラベル、Y軸ラベル、タイトルの追加
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')


def plot(X_train_std, X_test_std, y_train, y_test, y_pred, iris, model, cost):

    print('--- Train Data ---')
    print('Correctly classified train samples: %d' % (y_train == model.predict(X_train_std)).sum())
    print('Misclassified train samples: %d' % (y_train != model.predict(X_train_std)).sum())
    print('Train accuracy: %.2f' % accuracy_score(y_train, model.predict(X_train_std)))
    print('Train Cost: %.2f' % cost[0])

    print('--- Test Data ---')
    print('Correctly classified test samples: %d' % (y_test == y_pred).sum())
    print('Misclassified test samples: %d' % (y_test != y_pred).sum())
    print('Test accuracy: %.2f' % accuracy_score(y_test, y_pred))
    print('Test Cost: %.2f' % cost[1])

    colors = ['red', 'blue', 'green']
    markers = ['s', '^', 'o']
    cmap = ListedColormap(colors)
    class_names = iris.target_names

    # 決定境界の描画
    x1_min, x1_max = X_test_std[:, 0].min() - 1, X_test_std[:, 0].max() + 1
    x2_min, x2_max = X_test_std[:, 1].min() - 1, X_test_std[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),
                        np.arange(x2_min, x2_max, 0.02))
    Z = model.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.figure(figsize=(10, 7))
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for i, (marker, color, class_name) in enumerate(zip(markers, colors, class_names)):
        # correct
        correct_idx = np.where((y_test == i) & (y_pred == i))
        plt.scatter(x=X_test_std[correct_idx, 0],
                    y=X_test_std[correct_idx, 1],
                    marker=marker,
                    color=color,
                    edgecolor='black',
                    alpha=0.8,
                    label=f'Correct: {class_name}')
        # incorrectly classified
        incorrect_idx = np.where((y_test == i) & (y_pred != i))
        plt.scatter(x=X_test_std[incorrect_idx, 0],
                    y=X_test_std[incorrect_idx, 1],
                    marker=marker,
                    color=color,
                    edgecolor='yellow',
                    alpha=0.8,
                    label=f'Incorrect: {class_name}')

    plt.xlabel('Petal Length [standardized]')
    plt.ylabel('Petal Width [standardized]')
    plt.title('Classification Result on Test Set')
    plt.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig('classification_result.png')


def main():
    iris = datasets.load_iris()
    # 特徴量を2つに絞る
    X = iris.data[:, [2, 3]]
    # 目的変数
    y = iris.target
    df = pd.DataFrame(X, columns=iris.feature_names[2:])
    df['target'] = y
    df = preprocess(df)
    print(df.head())

    X_train_std, X_test_std, y_train, y_test = training_normalization(X, y)

    y_pred, model, cost = ppn(X_train_std, X_test_std, y_train, y_test)
    # y_pred, model, cost = logistic(X_train_std, X_test_std, y_train, y_test)
    # y_pred, model, cost = kmeans(X_train_std, X_test_std, y_train, y_test)
    # y_pred, model, cost = decision_tree(X_train_std, X_test_std, y_train, y_test)
    # y_pred, model, cost = random_forest(X_train_std, X_test_std, y_train, y_test)

    plot(X_train_std, X_test_std, y_train, y_test, y_pred, iris, model, cost)


if __name__ == "__main__":
    main()
