import pandas as pd

# Đọc dữ liệu OHLCB từ file CSV
df = pd.read_csv("data.csv")

# Kiểm tra các cột cần thiết
required_cols = ['Close', 'Open']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"File CSV phải có cột '{col}'.")

# Tính MA7 (trung bình động 7 phiên)
df['ma7'] = df['Close'].rolling(window=7).mean()

# Tính tỷ lệ (Close - ma7) / ma7
df['price_vs_ma7'] = (df['Close'] - df['ma7']) / df['ma7']

# Tính tỷ lệ (Close - Open) / Open
df['close/open'] = df['Close'] / df['Open'] - 1

df = df.dropna(subset=['ma7'])

# Lưu kết quả ra file mới (ghi đè)
df.to_csv("data.csv", index=False)

print("✅ Đã tính xong 'ma7', 'price_vs_ma7' và 'close_open_ratio'. File lưu tại: data.csv")
