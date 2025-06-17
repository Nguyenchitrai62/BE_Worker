import ccxt
import pandas as pd
from datetime import datetime

binance = ccxt.binance({
    'apiKey': '',
    'secret': '',
})

current_time = int(datetime.now().timestamp() * 1000)
print("Thời gian hiện tại:", current_time)

symbol = 'BTC/USDT'
limit = 1000  
total_limit = 50000  
num_requests = total_limit // limit  

ohlcv = []
for i in range(num_requests):
    data = binance.fetch_ohlcv(symbol, timeframe='1h', limit=limit, since=current_time - (i+1) * 1000 * 1000 * 60 * 60 * 1)
    if not data:
        break
    ohlcv[:0] = data
    print(f"{i+1} / {num_requests}")

df = pd.DataFrame(ohlcv, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])

from datetime import timedelta

df['Date'] = pd.to_datetime(df['Date'], unit='ms')

# # Cộng thêm 7 giờ để chuyển sang UTC+7
# df['Date'] = df['Date'] + timedelta(hours=7)

# # Định dạng ngắn gọn hơn: Tháng-Ngày-Năm Giờ:Phút
# df['Date'] = df['Date'].dt.strftime('%m/%d/%Y %H:%M')

df.to_csv('OHLCV.csv', index=False)

print("Dữ liệu đã được lưu vào file OHLCV.csv")
