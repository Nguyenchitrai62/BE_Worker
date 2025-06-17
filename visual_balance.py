import pandas as pd
import matplotlib.pyplot as plt

def evaluate_signals_with_plot(
    df, tp_percent=1.5, sl_percent=1.5, fee_percent=0.05, leverage=5, signal_col='signal', initial_balance=1.0
):
    """
    Mô phỏng giao dịch theo tín hiệu với lãi kép và đòn bẩy.
    Hỗ trợ cả tín hiệu mua (signal = 1) và tín hiệu bán (signal = 0).
    Thêm cột 'win' để đánh dấu kết quả từng lệnh (1 = win, 0 = loss).
    """
    required_cols = {'Open', 'High', 'Low', 'Close', 'Volume', signal_col}
    assert required_cols.issubset(df.columns), f"Thiếu cột: {required_cols - set(df.columns)}"

    # Tạo cột win với giá trị mặc định -1 (chưa có lệnh)
    df['win'] = -1
    
    wins = 0
    losses = 0
    total_signals = 0
    buy_signals = 0
    sell_signals = 0
    buy_wins = 0
    sell_wins = 0
    balance = initial_balance
    balance_history = []
    time_history = []

    i = 0
    while i < len(df):
        signal = df.loc[i, signal_col]
        
        # Tín hiệu mua (Long position)
        if signal == 1:
            entry_price = df.loc[i, 'Close']
            tp = entry_price * (1 + tp_percent / 100)
            sl = entry_price * (1 - sl_percent / 100)
            total_signals += 1
            buy_signals += 1

            for j in range(i + 1, len(df)):
                low = df.loc[j, 'Low']
                high = df.loc[j, 'High']

                # Lệnh bị hit SL
                if low < sl:
                    losses += 1
                    df.loc[i, 'win'] = 0  # Đánh dấu loss
                    loss_rate = (sl_percent + 2 * fee_percent) * leverage / 100
                    balance *= max(0, 1 - loss_rate)
                    balance_history.append(balance)
                    time_history.append(df.loc[j, 'timestamp'] if 'timestamp' in df.columns else j)
                    break
                # Lệnh hit TP
                elif high > tp:
                    wins += 1
                    buy_wins += 1
                    df.loc[i, 'win'] = 1  # Đánh dấu win
                    profit_rate = (tp_percent - 2 * fee_percent) * leverage / 100
                    balance *= (1 + profit_rate)
                    balance_history.append(balance)
                    time_history.append(df.loc[j, 'timestamp'] if 'timestamp' in df.columns else j)
                    break
        
        # Tín hiệu bán (Short position)
        elif signal == 0:
            entry_price = df.loc[i, 'Close']
            tp = entry_price * (1 - tp_percent / 100)  # TP thấp hơn giá vào lệnh
            sl = entry_price * (1 + sl_percent / 100)  # SL cao hơn giá vào lệnh
            total_signals += 1
            sell_signals += 1

            for j in range(i + 1, len(df)):
                low = df.loc[j, 'Low']
                high = df.loc[j, 'High']

                # Lệnh bị hit SL (giá tăng quá mức SL)
                if high > sl:
                    losses += 1
                    df.loc[i, 'win'] = 0  # Đánh dấu loss
                    loss_rate = (sl_percent + 2 * fee_percent) * leverage / 100
                    balance *= max(0, 1 - loss_rate)
                    balance_history.append(balance)
                    time_history.append(df.loc[j, 'timestamp'] if 'timestamp' in df.columns else j)
                    break
                # Lệnh hit TP (giá giảm xuống TP)
                elif low < tp:
                    wins += 1
                    sell_wins += 1
                    df.loc[i, 'win'] = 1  # Đánh dấu win
                    profit_rate = (tp_percent - 2 * fee_percent) * leverage / 100
                    balance *= (1 + profit_rate)
                    balance_history.append(balance)
                    time_history.append(df.loc[j, 'timestamp'] if 'timestamp' in df.columns else j)
                    break
        
        i += 1

    win_rate = (wins / total_signals * 100) if total_signals > 0 else 0.0
    buy_win_rate = (buy_wins / buy_signals * 100) if buy_signals > 0 else 0.0
    sell_win_rate = (sell_wins / sell_signals * 100) if sell_signals > 0 else 0.0
    total_return = (balance - initial_balance) / initial_balance * 100

    # Trả về cả thống kê và dữ liệu vẽ
    result = {
        'total_signals': total_signals,
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'wins': wins,
        'losses': losses,
        'buy_wins': buy_wins,
        'sell_wins': sell_wins,
        'win_rate': round(win_rate, 2),
        'buy_win_rate': round(buy_win_rate, 2),
        'sell_win_rate': round(sell_win_rate, 2),
        'final_balance': round(balance, 6),
        'total_return (%)': round(total_return, 2),
        'leverage': leverage,
        'balance_history': balance_history,
        'time_history': time_history
    }
    return result

# Đọc dữ liệu từ CSV
df = pd.read_csv('data.csv')

# Nếu không có timestamp thì tạo index datetime giả
# if 'timestamp' not in df.columns:
#     df['timestamp'] = pd.date_range(start='2022-01-01', periods=len(df), freq='H')

# Đánh giá và ghi lại kết quả
result = evaluate_signals_with_plot(df, tp_percent=1, sl_percent=1, fee_percent=0.05, leverage=1)

# In thống kê chi tiết
print("Kết quả đánh giá với lãi kép và đòn bẩy:")
print(f"Đòn bẩy: x{result['leverage']}")
print(f"Tổng tín hiệu: {result['total_signals']}")
print(f"  - Tín hiệu mua (Long): {result['buy_signals']}")
print(f"  - Tín hiệu bán (Short): {result['sell_signals']}")
print(f"Win: {result['wins']}, Loss: {result['losses']}")
print(f"  - Buy wins: {result['buy_wins']}")
print(f"  - Sell wins: {result['sell_wins']}")
print(f"Tỉ lệ thắng tổng: {result['win_rate']}%")
print(f"  - Tỉ lệ thắng Buy: {result['buy_win_rate']}%")
print(f"  - Tỉ lệ thắng Sell: {result['sell_win_rate']}%")
print(f"Số vốn cuối cùng: {result['final_balance']}x")
print(f"Tổng lợi nhuận: {result['total_return (%)']}%")

# In thông tin về cột win
print(f"\nThông tin cột 'win':")
print(f"- Số lệnh win (win=1): {len(df[df['win']==1])}")
print(f"- Số lệnh loss (win=0): {len(df[df['win']==0])}")
print(f"- Số dòng không có lệnh (win=-1): {len(df[df['win']==-1])}")

# # Hiển thị một vài dòng có signal để kiểm tra
# print(f"\nMột vài dòng có signal và kết quả:")
# signals_df = df[df[result.get('signal', 'signal')] != -1][['Open', 'High', 'Low', 'Close', 'signal', 'win']].head(10)
# print(signals_df)

# Lưu DataFrame với cột win mới
df.to_csv('data.csv', index=False)
print(f"\nĐã lưu DataFrame với cột 'win' vào file: data.csv")

# Vẽ biểu đồ tăng trưởng vốn (giữ nguyên 1 đồ thị)
plt.figure(figsize=(12, 6))
plt.plot(result['time_history'], result['balance_history'], label='Balance Growth', color='blue')
plt.axhline(1.0, color='gray', linestyle='--', label='Initial Balance')
plt.title('Tăng trưởng số dư theo thời gian')
plt.xlabel('Thời gian')
plt.ylabel('Số dư (x lần ban đầu)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()