import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.fft as fft
from scipy import signal

# ==========================================
# 1. データ準備（ダミーデータ生成）
# ==========================================
# 実際のデータを使う場合は df = pd.read_csv(...) 等に置き換えてください
np.random.seed(42)
days = 1000
date_range = pd.date_range(start='2020-01-01', periods=days, freq='B') # 営業日

# 周期的なボラティリティ変動を模擬
# ベースのノイズ + 20日周期(約1ヶ月) + 5日周期(1週間) の成分を入れる
t = np.arange(days)
periodic_vola = 1 + 0.5 * np.sin(2 * np.pi * t / 20) + 0.3 * np.sin(2 * np.pi * t / 5)
returns = np.random.normal(0, periodic_vola * 0.01) # 標準偏差が周期的に変動

df = pd.DataFrame({'Close': 100 * np.cumprod(1 + returns)}, index=date_range)

# ==========================================
# 2. 前処理: ボラティリティ・プロキシの算出
# ==========================================
# 対数リターン
df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
df = df.dropna()

# ボラティリティのプロキシとして「絶対値リターン」を使用
# (二乗リターンよりも外れ値の影響を受けにくいため、周期解析にはこちらが推奨されることが多い)
volatility_proxy = np.abs(df['Log_Return']).values

# 平均値を引く (DC成分の除去)
volatility_detrended = volatility_proxy - np.mean(volatility_proxy)

# 窓関数の適用 (スペクトル漏れの抑制) - Hanning窓を使用
window = np.hanning(len(volatility_detrended))
volatility_windowed = volatility_detrended * window

# ==========================================
# 3. FFT (高速フーリエ変換) の実行
# ==========================================
N = len(volatility_windowed)
dt = 1  # サンプリング間隔 (1日)

# rfft: 実数データのFFT (正の周波数のみ返す)
fft_vals = fft.rfft(volatility_windowed)
freqs = fft.rfftfreq(N, d=dt)

# パワースペクトル密度 (PSD) の計算: |FFT|^2
power = np.abs(fft_vals)**2

# ==========================================
# 4. 可視化 (Periodogram)
# ==========================================
fig, axes = plt.subplots(2, 1, figsize=(12, 10), constrained_layout=True)

# --- 上段: ボラティリティ時系列 ---
axes[0].plot(df.index, volatility_proxy, color='grey', alpha=0.6, label='|Return|')
axes[0].set_title('Volatility Proxy (Absolute Returns)', fontsize=14)
axes[0].set_ylabel('Absolute Return')
axes[0].grid(True, linestyle='--')

# --- 下段: パワースペクトル ---
# ※ 周期 (Period = 1/Frequency) でプロット
valid_idx = freqs > 0 # 0Hz (DC成分) を除外
periods = 1 / freqs[valid_idx]
power_valid = power[valid_idx]

axes[1].plot(periods, power_valid, color='darkblue', lw=1.5)
axes[1].set_title('Power Spectrum of Volatility (Periodogram)', fontsize=14)
axes[1].set_xlabel('Period [Days]', fontsize=12)
axes[1].set_ylabel('Power (Arbitrary Units)', fontsize=12)

# X軸を対数スケールにすると、広い範囲の周期が見やすい
axes[1].set_xscale('log')
axes[1].set_xlim(2, N/2) # ナイキスト周波数(2日) 〜 観測期間の半分程度までを表示
axes[1].grid(True, which="both", ls="-", alpha=0.5)

# ピーク検出と注釈（上位3つのピークを表示）
peaks, _ = signal.find_peaks(power_valid, distance=5) # distanceは近傍ピーク除去用
top_peaks_idx = peaks[np.argsort(power_valid[peaks])[-3:][::-1]] # 上位3つ

for idx in top_peaks_idx:
    peak_period = periods[idx]
    peak_power = power_valid[idx]
    axes[1].plot(peak_period, peak_power, "ro")
    axes[1].annotate(f'{peak_period:.1f} Days',
                     xy=(peak_period, peak_power),
                     xytext=(peak_period, peak_power*1.1),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     fontsize=10, ha='center')

plt.tight_layout()
plt.show()
