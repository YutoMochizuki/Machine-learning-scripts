import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. データと設定の準備
# ==========================================
# プロットのスタイル設定（論文向けの視認性確保）
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# df_testは事前に用意されたDataFrameと仮定
# colsはプロット対象の数値カラム名リストと仮定
# 例:
df_test = pd.read_csv('your_data.csv')
cols = ['feature1', 'feature2', 'feature3', 'feature4']
# --------------------------------------------------

# ==========================================
# 2. コーナープロット (Scatter Plot Matrix)
# ==========================================
# mlxtendの代わりにseabornのpairplotを使用し、heatmapとスタイルを統一
# corner=Trueにすることで、冗長な右上半分をカットし、対角線に分布図を表示
g = sns.pairplot(
    df_test[cols],
    diag_kind='kde',      # 対角線はカーネル密度推定（ヒストグラムなら'hist'）
    kind='scatter',       # 散布図
    corner=True,          # 左下のみ表示（論文等で好まれるスタイル）
    plot_kws={'alpha': 0.5, 's': 30, 'edgecolor': 'k', 'linewidth': 0.5}, # プロットの装飾
    diag_kws={'fill': True, 'alpha': 0.3}, # 分布図の装飾
    height=2.5            # 1つあたりの図のサイズ
)

g.fig.suptitle('Scatter Plot Matrix (Corner Plot)', y=1.02) # タイトル調整


# ==========================================
# 3. 相関行列のヒートマップ
# ==========================================
# 相関行列の計算 (Pandasのメソッドを使用する方がラベル管理が安全)
corr_matrix = df_test[cols].corr()

plt.figure(figsize=(10, 8))

# マスク作成（対角線より上を隠す場合。全て表示したい場合はmask引数を削除）
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

hm = sns.heatmap(
    corr_matrix,
    mask=mask,            # 右上を隠す（コーナープロットと視覚的に合わせるため）
    cmap='coolwarm',      # 青-赤のカラーマップ
    center=0,             # 0を白（中間色）にする
    vmax=1, vmin=-1,      # 範囲を固定
    annot=True,           # 数値表示
    fmt='.2f',            # 桁数指定
    square=True,          # 正方形を維持
    linewidths=.5,        # グリッド線
    cbar_kws={"shrink": .8} # カラーバーのサイズ調整
)

plt.title('Correlation Matrix Heatmap', fontsize=16, y=1.02)
plt.tight_layout()

# ==========================================
# 4. プロットの表示
# ==========================================
plt.show()
