import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model  # pip install arch
import seaborn as sns

# ==========================================
# 0. 設定と物理的背景
# ==========================================
# GARCH(1,1) プロセス:
# r_t = sigma_t * epsilon_t
# sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2
#
# - omega: ベースラインの分散 (定数項)
# - alpha (ARCH項): 直近のショックへの感度 (突発的な変動への反応)
# - beta (GARCH項): 過去のボラティリティの記憶 (持続性/Persistence)
#
# 安定条件: alpha + beta < 1 (これを超えると分散が発散する)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
np.random.seed(42)

# ==========================================
# 1. データ生成 (GARCH過程のシミュレーション)
# ==========================================
n = 2000
omega_true = 0.1
alpha_true = 0.1  # 前日の変動にどれだけ反応するか
beta_true = 0.8   # 変動がどれだけ後を引くか (減衰時定数に対応)

returns = np.zeros(n)
sigma_sq = np.zeros(n)
epsilon = np.random.normal(0, 1, n)

# 初期化
sigma_sq[0] = omega_true / (1 - alpha_true - beta_true) # 長期平均分散

# 時間発展 (Time Evolution)
for t in range(1, n):
    # 条件付き分散の更新式 (Equation of Motion for Variance)
    sigma_sq[t] = omega_true + alpha_true * (returns[t-1]**2) + beta_true * sigma_sq[t-1]
    # 観測されるリターン
    returns[t] = np.sqrt(sigma_sq[t]) * epsilon[t]

# DataFrame化
dates = pd.date_range(start='2018-01-01', periods=n, freq='B')
df = pd.DataFrame({'Return': returns}, index=dates)

# ==========================================
# 2. モデル構築と推定 (MLE: 最尤推定法)
# ==========================================
# リターン系列のみを入力とする（ボラティリティは隠れ変数として推定される）
# vol='Garch', p=1, q=1 は GARCH(1,1) を指定
model = arch_model(df['Return'], vol='Garch', p=1, q=1, dist='Normal')

# フィッティング
res = model.fit(update_freq=0, disp='off')

# 結果の表示
print(res.summary())

# ==========================================
# 3. 推定結果の可視化
# ==========================================
fig, ax = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# (a) リターン時系列
ax[0].plot(df.index, df['Return'], color='grey', alpha=0.5, label='Daily Returns')
ax[0].set_title('Observed Returns (Simulated GARCH Process)', fontsize=14)
ax[0].set_ylabel('Return')
ax[0].legend(loc='upper right')

# (b) 推定された条件付きボラティリティ vs 真のボラティリティ
# res.conditional_volatility で推定された sigma_t が取得可能
estimated_vol = res.conditional_volatility

ax[1].plot(df.index, np.sqrt(sigma_sq), color='black', linewidth=2, label='True Volatility $\sigma_t$')
ax[1].plot(df.index, estimated_vol, color='red', linestyle='--', linewidth=1.5, label='Estimated GARCH $\hat{\sigma}_t$')
ax[1].set_title('Conditional Volatility: True vs Estimated', fontsize=14)
ax[1].set_ylabel('Volatility ($\sigma$)')
ax[1].legend(loc='upper right')

plt.tight_layout()
plt.show()

# ==========================================
# 4. パラメータの物理的解釈
# ==========================================
print("\n--- Parameter Interpretation ---")
params = res.params
persistence = params['alpha[1]'] + params['beta[1]']
print(f"Alpha (Shock Sensitivity): {params['alpha[1]']:.4f}")
print(f"Beta (Persistence):        {params['beta[1]']:.4f}")
print(f"Persistence (Alpha+Beta):  {persistence:.4f}")
print(f"Half-life of Volatility:   {np.log(0.5) / np.log(persistence):.2f} days")

if persistence >= 1:
    print("Warning: The process is non-stationary (IGARCH).")
else:
    print("Note: The process is mean-reverting.")
