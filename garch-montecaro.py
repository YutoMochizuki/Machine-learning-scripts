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
# last_obsを指定して、最後の30日分を「未来」として残し、予測性能評価用にすることも可能だが
# ここでは全データを使って学習し、その先を予測する設定とする。
res = model.fit(update_freq=0, disp='off')

# 結果の表示
print(res.summary())

# ==========================================
# 3. 推定結果の可視化 (過去のフィッティング)
# ==========================================
fig, ax = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# (a) リターン時系列
ax[0].plot(df.index, df['Return'], color='grey', alpha=0.5, label='Daily Returns')
ax[0].set_title('Observed Returns (Simulated GARCH Process)', fontsize=14)
ax[0].set_ylabel('Return')
ax[0].legend(loc='upper right')

# (b) 推定された条件付きボラティリティ vs 真のボラティリティ
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
unconditional_var = params['omega'] / (1 - persistence)

print(f"Alpha (Shock Sensitivity): {params['alpha[1]']:.4f}")
print(f"Beta (Persistence):        {params['beta[1]']:.4f}")
print(f"Persistence (Alpha+Beta):  {persistence:.4f}")
print(f"Half-life of Volatility:   {np.log(0.5) / np.log(persistence):.2f} days")
print(f"Long-run Variance (Uncond): {unconditional_var:.4f}")

# ==========================================
# 5. 将来予測 (Forecasting)
# ==========================================
# GARCH予測の核心:
# 短期的には現在のボラティリティに依存するが、長期的には「無条件分散（長期平均）」に収束していく。
# 予測期間: 向こう30日間
horizon = 30

# --- A. 解析的予測 (Analytical Forecast) ---
# 期待値としての分散の予測
forecasts = res.forecast(horizon=horizon, reindex=False)
# 最終観測日からの予測値を取得 (h.1, h.2, ... h.30)
var_forecast = forecasts.variance.iloc[-1]
vol_forecast = np.sqrt(var_forecast) # 標準偏差（ボラティリティ）に変換

# --- B. シミュレーション予測 (Monte Carlo Simulation) ---
# 分布の広がり（信頼区間）を知るためにパスを多数生成
sim_forecasts = res.forecast(horizon=horizon, method='simulation', simulations=1000, reindex=False)
sim_paths = sim_forecasts.simulations.values[-1, :, :] # shape: (simulations, horizon)
# ボラティリティパスに変換 (リターンの分散ではなく、ボラティリティ自体のパスが必要な場合、
# archのsimulationは「リターンのパス」を返すため、そこから分散を再計算する必要があるが、
# ここでは簡易的に「解析的予測分散」を中心に、リターンの広がりを見るファンチャートを作成する)

# 注: archの forecast(method='simulation') は「将来のリターン」の乱数パスを返す。
# 将来の「ボラティリティ」の不確実性を見たい場合、各パスでGARCH式を回す必要があるが、
# ライブラリの仕様上、リターンの累積分布を見るのが一般的。

# ここでは「ボラティリティの平均回帰」を視覚化するためのプロットを作成
plt.figure(figsize=(12, 6))

# X軸: 予測期間
future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=horizon, freq='B')

# 1. 長期平均水準（Unconditional Volatility）
long_run_vol = np.sqrt(unconditional_var)
plt.axhline(long_run_vol, color='green', linestyle=':', linewidth=2, label=f'Long-run Average Volatility ({long_run_vol:.2f})')

# 2. 解析的予測パス
plt.plot(future_dates, vol_forecast, color='red', marker='o', markersize=4, label='Analytical Forecast')

# 3. 直近の実績値（接続用）
last_obs_date = df.index[-50:]
last_obs_vol = estimated_vol[-50:]
plt.plot(last_obs_date, last_obs_vol, color='black', alpha=0.7, label='Historical Volatility (Last 50 days)')

plt.title('GARCH Volatility Forecast: Mean Reversion to Long-run Average', fontsize=14)
plt.ylabel('Volatility')
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.show()

# 解説
print("\n--- Forecast Interpretation ---")
print("The forecast shows the 'Mean Reversion' property of GARCH models.")
if vol_forecast.iloc[0] > long_run_vol:
    print("Current volatility is HIGH. Ideally, it is expected to DECREASE towards the long-run average.")
else:
    print("Current volatility is LOW. Ideally, it is expected to INCREASE towards the long-run average.")
print(f"Convergence speed is determined by (Alpha + Beta) = {persistence:.4f}")
