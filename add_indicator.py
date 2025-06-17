import pandas as pd
import ta
import numpy as np

# Đọc dữ liệu
df = pd.read_csv('data.csv', parse_dates=['Date'])
df = df.sort_values('Date')

# ====== Giá cơ bản ======
df['close/open'] = df['Close'] / df['Open'] - 1
df['high-low'] = (df['High'] - df['Low']) / df['Close']

# ====== EMA + MACD ======
ema12 = ta.trend.ema_indicator(close=df['Close'], window=12)
ema26 = ta.trend.ema_indicator(close=df['Close'], window=26)
df['ema12'] = ema12 / df['Close'] - 1
df['ema26'] = ema26 / df['Close'] - 1
df['macd'] = df['ema12'] - df['ema26']
df['ema12_slope'] = ema12.diff() / ema12.shift(1)  # độ dốc ema12
df['macd_hist'] = ta.trend.macd_diff(close=df['Close'])  # histogram MACD gốc

# ====== RSI / StochRSI ======
df['rsi14'] = ta.momentum.rsi(close=df['Close'], window=14) / 50 - 1
df['stoch_rsi'] = ta.momentum.stochrsi(close=df['Close'], window=14) * 2 - 1

# ====== Bollinger Bands ======
bb = ta.volatility.BollingerBands(close=df['Close'], window=14)
df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
df['bb_position'] = (df['Close'] - bb.bollinger_mavg()) / (bb.bollinger_hband() - bb.bollinger_lband())
df['bb_position'] = df['bb_position'].clip(-1, 1)

# ====== ATR ======
df['atr'] = ta.volatility.average_true_range(high=df['High'], low=df['Low'], close=df['Close'], window=14) / df['Close']

# ====== OBV và độ dốc OBV ======
obv = ta.volume.on_balance_volume(close=df['Close'], volume=df['Volume'])
df['obv_slope'] = obv.diff() / obv.shift(1)

# ====== Recent Volatility ======
df['recent_volatility'] = df['Close'].pct_change().rolling(window=5).std()

# Xoá dòng NaN (các chỉ báo dùng rolling/window sẽ sinh NaN ban đầu)
df = df.dropna(subset=['macd_hist'])

# Xuất file kết quả
df.to_csv('data.csv', index=False)

# In thử
print(df[[
    'ema12_slope', 'macd_hist', 'rsi14', 'stoch_rsi',
    'bb_width', 'bb_position', 'atr', 'obv_slope', 'recent_volatility'
]].head(3))
