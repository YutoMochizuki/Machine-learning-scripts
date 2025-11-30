import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. データ準備 (ダミーデータの生成)
# ---------------------------------------------------------
# 実際の解析では、ご自身の高頻度データから計算した日次RVを使用してください。
np.random.seed(42)
n_days = 1000

# 幾何ブラウン運動でRVのような時系列を模擬生成
# RVは対数正規分布に近い挙動を示すことが多いため
dt = 1
mu = 0.0
sigma = 0.5
random_shocks = np.random.normal(0, 1, n_days)
log_rv_sim = np.zeros(n_days)
log_rv_sim[0] = 0

for t in range(1, n_days):
    # 平均回帰性を持たせる (Ornstein-Uhlenbeck過程風)
    log_rv_sim[t] = log_rv_sim[t-1] + 0.1 * (mu - log_rv_sim[t-1]) * dt + sigma * np.sqrt(dt) * random_shocks[t]

# 対数から実数空間(RV)に戻す
rv_data = np.exp(log_rv_sim)
dates = pd.date_range(start='2020-01-01', periods=n_days, freq='B') # 営業日

df = pd.DataFrame({'RV': rv_data}, index=dates)

# ---------------------------------------------------------
# 2. 特徴量エンジニアリング (HARモデルの構築)
# ---------------------------------------------------------
# 論文の式(8)と(68)に基づく実装 [cite: 66, 68]
# log RV_t = alpha + beta_d * log RV_{t-1} + beta_w * log RV_{t-5:t-1} + beta_m * log RV_{t-22:t-1} + error

# まず、RVの対数をとる (分布を正規分布に近づけるため)
df['log_RV'] = np.log(df['RV'])

# 各期間の成分を作成 (Shiftを使って「予測時点から見た過去」にする)
# 日次成分 (Daily): 1日前のlog RV
df['log_RV_d'] = df['log_RV'].shift(1)

# 週次成分 (Weekly): 過去5日間のRVの平均の対数
# 注意: 論文の定義 Eq(68) では「RVの平均」をとってから「対数」をとる形式が示唆されています。
# RV_{t-5:t-1} = (1/5) * sum(RV_{t-i})
df['RV_w_raw'] = df['RV'].shift(1).rolling(window=5).mean()
df['log_RV_w'] = np.log(df['RV_w_raw'])

# 月次成分 (Monthly): 過去22日間のRVの平均の対数
df['RV_m_raw'] = df['RV'].shift(1).rolling(window=22).mean()
df['log_RV_m'] = np.log(df['RV_m_raw'])

# 欠損値（最初の22日分）を除去
df_model = df.dropna().copy()

# ---------------------------------------------------------
# 3. モデル推定 (OLS: 最小二乗法)
# ---------------------------------------------------------
# 目的変数: 当日の log RV
Y = df_model['log_RV']

# 説明変数: 定数項 + 日次 + 週次 + 月次
X = df_model[['log_RV_d', 'log_RV_w', 'log_RV_m']]
X = sm.add_constant(X) # 定数項(alpha)を追加

# OLSでフィッティング
model = sm.OLS(Y, X).fit()

# 結果の表示
print(model.summary())

# ---------------------------------------------------------
# 4. 予測と可視化
# ---------------------------------------------------------
# モデルによる予測値 (対数空間)
df_model['pred_log_RV'] = model.predict(X)

# 実空間に戻す (exp)
# 注: 単純なexpではバイアスが生じる場合がありますが、ここでは簡易的に変換します。
# 厳密には論文式(29)にあるような分散補正項 exp(pred + 0.5*sigma^2) を考慮します。
resid_var = model.mse_resid
df_model['pred_RV'] = np.exp(df_model['pred_log_RV'] + 0.5 * resid_var)

# プロット
plt.figure(figsize=(12, 6))
# 視認性を良くするため、直近100日分を表示
subset = df_model.iloc[-100:]

plt.plot(subset.index, subset['RV'], label='Actual RV', color='black', alpha=0.6)
plt.plot(subset.index, subset['pred_RV'], label='HAR Forecast', color='red', linestyle='--')
plt.title('HAR Model Forecast vs Actual (Last 100 Days)')
plt.ylabel('Realized Volatility')
plt.legend()
plt.grid(True)
plt.show()

# 係数の確認 (物理的解釈: 各タイムスケールの寄与度)
print("\n--- Estimated Parameters ---")
print(model.params)
