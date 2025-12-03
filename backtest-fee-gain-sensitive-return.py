import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 設定セクション (Configuration)
# ==========================================
ENABLE_COST = True       # 手数料コストの有無
COST_RATE   = 0.001      # コスト率 (0.1%)

# 【重要】可変モデルのパラメータ
GAIN        = 50.0       # 感度 (Gain): 予測リターン1%に対してポジション何%を持つか
                         # 例: 50.0なら、予測+1%でポジション0.5(50%)、予測+2%で1.0(100%)
ALLOW_SHORT = True      # 空売り(ショート)を許可するか (True: -1.0~1.0, False: 0.0~1.0)

# ==========================================
# 1. データ生成 (Data Generation)
# ==========================================
np.random.seed(42)
days = 300
dates = pd.date_range(start='2024-01-01', periods=days)
prices = 100 + np.cumsum(np.random.randn(days))
df = pd.DataFrame(data={'Close': prices}, index=dates)

# 予測データの生成
# 今回はシグナルの強弱を見るため、ノイズを少し大きくして変動を作ります
noise = np.random.normal(0, 0.8, days)
df['Predicted_Next_Close'] = df['Close'].shift(-1) + noise

# ==========================================
# 2. 戦略ロジック: 可変シグナル (Proportional Signal)
# ==========================================
# A. 予測リターン (Alpha) の計算: (予測 - 現在) / 現在
df['Predicted_Return'] = (df['Predicted_Next_Close'] - df['Close']) / df['Close']

# B. 生シグナル (Raw Signal) = Alpha × Gain
df['Raw_Signal'] = df['Predicted_Return'] * GAIN

# C. クリッピング (Saturation)
# ポジションサイズを制限します (リスク管理)
# 規格化するわけではない。MAXを超えたものをMAXに制限するだけ。
if ALLOW_SHORT:
    # ショート許可: -1.0 (全力売り) 〜 1.0 (全力買い)
    df['Signal'] = df['Raw_Signal'].clip(lower=-1.0, upper=1.0)
else:
    # ロングのみ: 0.0 (ノーポジ) 〜 1.0 (全力買い)
    df['Signal'] = df['Raw_Signal'].clip(lower=0.0, upper=1.0)

# --- 以下、収益計算 ---

# 市場リターン
df['Actual_Return'] = df['Close'].pct_change()

# 戦略リターン (Gross)
# ポジション量(0.0〜1.0) に応じてリターンも比例配分される
df['Gross_Strategy_Return'] = df['Signal'].shift(1) * df['Actual_Return']

# ==========================================
# 3. コスト計算 (Cost Calculation)
# ==========================================
# 可変モデルでは「微調整」が頻発するため、ここが重要になります
# 差分 (0.8 -> 0.9 の場合、0.1分のコストがかかる)
pos_change = df['Signal'].diff().abs().fillna(0)

if ENABLE_COST:
    df['Transaction_Cost'] = pos_change * COST_RATE
else:
    df['Transaction_Cost'] = 0.0

# 純リターン (Net)
df['Net_Strategy_Return'] = df['Gross_Strategy_Return'] - df['Transaction_Cost']

# 累積収益
df['Cum_Market'] = (1 + df['Actual_Return']).cumprod()
df['Cum_Strategy'] = (1 + df['Net_Strategy_Return']).cumprod()

# ==========================================
# 4. 可視化 (Visualization - 3 Panels)
# ==========================================
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1, 2]})

# --- 上段: 価格推移 ---
ax1.plot(df.index, df['Close'], label='Actual Close', color='black', alpha=0.6)
ax1.plot(df.index, df['Predicted_Next_Close'].shift(1), label='Predicted (Shifted)',
         color='orange', linestyle='--', alpha=0.8)
ax1.set_title('1. Price: Actual vs Predicted')
ax1.set_ylabel('Price')
ax1.legend(loc='upper left')
ax1.grid(True)

# --- 中段: ポジション量の推移 (ここが可変モデルの肝) ---
# 0.0〜1.0の間でどう動いているかを表示
ax2.fill_between(df.index, df['Signal'], 0, color='green', alpha=0.3, label='Long Position Size')
if ALLOW_SHORT:
    ax2.fill_between(df.index, df['Signal'], 0, where=(df['Signal'] < 0), color='red', alpha=0.3, label='Short Position Size')
    ax2.set_ylim(-1.1, 1.1)
else:
    ax2.set_ylim(-0.1, 1.1)

ax2.set_title(f'2. Position Sizing (Gain={GAIN}, Short={ALLOW_SHORT})')
ax2.set_ylabel('Size (0.0 - 1.0)')
ax2.grid(True)

# --- 下段: 累積収益率 ---
cost_str = "ON" if ENABLE_COST else "OFF"
ax3.plot(df.index, df['Cum_Market'], label='Market (Buy & Hold)', linestyle='--', color='gray')
ax3.plot(df.index, df['Cum_Strategy'], label=f'Strategy (Cost {cost_str})', color='blue', linewidth=2)
ax3.set_title('3. Cumulative Return')
ax3.set_ylabel('Equity Multiplier')
ax3.legend(loc='upper left')
ax3.grid(True)

plt.tight_layout()
plt.show()

# ==========================================
# 結果サマリー
# ==========================================
print(f"--- Strategy Report (Gain: {GAIN}) ---")
print(f"総取引回数: {pos_change.sum():.0f} 回")
print(f"最終リターン (Market): {df['Cum_Market'].iloc[-1] - 1:.2%}")
print(f"最終リターン (Strategy): {df['Cum_Strategy'].iloc[-1] - 1:.2%}")
