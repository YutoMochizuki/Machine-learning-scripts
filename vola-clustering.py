import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
import seaborn as sns

# ==========================================
# 1. 設定とデータ準備
# ==========================================
sns.set_theme(style="ticks", context="paper", font_scale=1.2)
plt.rcParams['figure.constrained_layout.use'] = True

# --- [ダミーデータの生成] ---
np.random.seed(42)
n_samples = 5000
sigma = np.ones(n_samples)
returns = np.zeros(n_samples)
for t in range(1, n_samples):
    # GARCH(1,1)風の簡易ロジック
    sigma[t] = np.sqrt(0.01 + 0.85 * sigma[t-1]**2 + 0.1 * returns[t-1]**2)
    returns[t] = np.random.normal(0, sigma[t])

df = pd.DataFrame({'Return': returns})

# 前処理
returns_norm = (df['Return'] - df['Return'].mean()) / df['Return'].std()
absolute_returns = np.abs(returns_norm)

# ==========================================
# 2. 関数定義: レバレッジ効果の計算
# ==========================================
def calculate_leverage_effect(norm_returns, max_lag=50):
    returns_arr = norm_returns.values
    vals = []
    # 分母の正規化定数 Z = <r^2>^2
    # 入力が正規化(分散1)されていれば、E_r2 ≈ 1 なので denominator ≈ 1 です。
    # 無次元量にするための規格化です。
    E_r2 = np.mean(returns_arr**2)
    denominator = E_r2**2

    for k in range(1, max_lag + 1):
        # r(t): 時刻 t の「符号付き」リターン（変位）
        # 配列の最後 k 個を除外（ペアを作る相手がいないため）
        r_t = returns_arr[:-k]

        # r(t+k)^2: 時刻 t+k の「二乗」リターン（エネルギー/振幅の大きさ）
        # 配列の最初 k 個を除外（ペアを作る相手がいないため）
        r_t_plus_k_sq = returns_arr[k:]**2

        # 分子: < r(t) * r(t+k)^2 >
        # 「今の価格変動の向き」と「未来のボラティリティの大きさ」の積の平均
        numerator = np.mean(r_t * r_t_plus_k_sq)

        # L(k) を計算
        L_k = numerator / denominator
        vals.append(L_k)

    return np.array(vals)

# ==========================================
# 3. プロット生成
# ==========================================
fig = plt.figure(figsize=(18, 5))

# --- (a) リターンの自己相関 (Linear Autocorrelation) ---
# 自己相関が95%信頼区間内に収まることを確認→収まっていたら、市場は効率的でランダムウォーク仮説を支持
ax1 = fig.add_subplot(131)
# 修正点: auto_ylims=True を削除しました
plot_acf(df["Return"], lags=30, ax=ax1, zero=False,
         title="Fig 1(a): Autocorrelation of Returns")
ax1.set_xlabel("Lag (k)")
ax1.set_ylabel("Autocorrelation")
ax1.grid(True, linestyle='--', alpha=0.6)

# --- (b) ボラティリティ・クラスタリング (Volatility Clustering) ---
# 自己相関関数を対数-対数スケールでプロットし、べき乗則に従うことを確認
# ラグが大きくなるにつれてゆっくりと減衰することを示す→ボラティリティの持続性を示唆
ax2 = fig.add_subplot(132)
lags_vol = 500
acf_abs = acf(absolute_returns, nlags=lags_vol, fft=True)

ax2.plot(range(1, len(acf_abs)), acf_abs[1:], 'o', markersize=2, color='darkblue', alpha=0.6)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel("Lag (k) [Log Scale]")
ax2.set_ylabel("Autocorrelation of |r| [Log Scale]")
ax2.set_title("Fig 1(b): Volatility Clustering (Power Law)")
ax2.grid(True, which="both", linestyle='--', alpha=0.4)

# --- (c) レバレッジ効果 (Leverage Effect) ---
ax3 = fig.add_subplot(133)
max_lag_lev = 50
L_values = calculate_leverage_effect(returns_norm, max_lag=max_lag_lev)

ax3.plot(range(1, max_lag_lev + 1), L_values, 'o-', markersize=4, color='crimson', label='$L(k)$')
ax3.axhline(0, color='black', linestyle='-', linewidth=0.8)
ax3.set_xlabel("Lag (k)")
ax3.set_ylabel("Correlation $L(k)$")
ax3.set_title("Fig 1(c): Leverage Effect")
ax3.legend()
ax3.grid(True, linestyle='--', alpha=0.6)

# 保存と表示
plt.tight_layout()
plt.savefig("Financial_Stylized_Facts.png", dpi=300, bbox_inches='tight')
plt.show()
