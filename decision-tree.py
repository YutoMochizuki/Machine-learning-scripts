import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression

# ==========================================
# 1. データ生成（ダミーデータ）
# ==========================================
# 20個の特徴量のうち、本当に効いているのは10個だけとする
X, y = make_regression(n_samples=200, n_features=20, n_informative=10, noise=0.5, random_state=42)
feature_names = np.array([f'Feature_{i}' for i in range(X.shape[1])])

# Lassoはスケーリングが必須（係数の大きさを公平に評価するため）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 2. Lassoによる特徴量選択
# ==========================================
# alphaでスパース性を制御（大きくするとより多くの変数が0になり消える）
lasso = Lasso(alpha=0.3, random_state=42)
lasso.fit(X_scaled, y)

# 係数が0でない（生き残った）特徴量を抽出
mask = lasso.coef_ != 0
X_selected = X[:, mask]  # 選択された特徴量だけのデータ
selected_names = feature_names[mask]

print(f"元の特徴量数: {len(feature_names)}")
print(f"Lasso選択後: {len(selected_names)}")
print(f"選択された特徴量: {selected_names}")

# ==========================================
# 3. 決定木による可視化（Surrogate Model）
# ==========================================
# Lassoで選ばれた特徴量だけを使って、シンプルな決定木を学習させる
# max_depthを浅くする（2~3）のがコツ。深すぎると図が読めなくなる。
tree_model = DecisionTreeRegressor(max_depth=3, random_state=42)
tree_model.fit(X_selected, y)

# 描画
plt.figure(figsize=(24, 12))  # かなり大きめに確保

# 3. プロット（fontsizeを極端に大きくするのがコツ）
plot_tree(
    tree_model,
    feature_names=selected_names,
    filled=True,
    rounded=True,
    precision=2,
    # 【ここが重要】フォントを大きくすると、箱も大きくなり全体に広がります
    fontsize=15,
    proportion=True
)

plt.title("Decision Tree Visualization using Lasso-Selected Features")
plt.savefig("decision_tree_lasso_features.png")
