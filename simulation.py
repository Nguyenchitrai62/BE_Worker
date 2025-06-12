import pandas as pd
import matplotlib.pyplot as plt

def evaluate_signals_realistic(
    df, tp_percent=1.5, sl_percent=1.5, fee_percent=0.05, leverage=5, 
    signal_col='signal', initial_balance=1.0, max_orders=5, position_size=0.2,
    account_stop_loss=0.5
):
    """
    Mô phỏng giao dịch thực tế:
    - Duyệt chuỗi 1 lần duy nhất
    - Mỗi lệnh vào position_size (20%) vốn ban đầu (cố định)
    - Tối đa max_orders (5) lệnh cùng lúc
    - Dừng giao dịch nếu tài khoản < account_stop_loss * vốn ban đầu
    - Hỗ trợ cả Long (signal=1) và Short (signal=0)
    """
    required_cols = {'Open', 'High', 'Low', 'Close', 'Volume', signal_col}
    assert required_cols.issubset(df.columns), f"Thiếu cột: {required_cols - set(df.columns)}"

    # Thống kê
    wins = 0
    losses = 0
    total_signals = 0
    buy_signals = 0
    sell_signals = 0
    buy_wins = 0
    sell_wins = 0
    
    # Quản lý tài khoản
    balance = initial_balance
    balance_history = [balance]
    time_history = [0]
    fixed_order_amount = initial_balance * position_size  # Số tiền cố định mỗi lệnh
    account_blown = False  # Flag kiểm tra tài khoản bị thổi
    
    # Quản lý các lệnh đang mở
    active_orders = []  # List các dict chứa thông tin lệnh
    
    # Duyệt qua từng candle
    for i in range(len(df)):
        # Kiểm tra stop loss tài khoản
        if balance < initial_balance * account_stop_loss:
            account_blown = True
            break
            
        current_time = df.loc[i, 'Date'] if 'Date' in df.columns else i
        current_high = df.loc[i, 'High']
        current_low = df.loc[i, 'Low']
        current_close = df.loc[i, 'Close']
        signal = df.loc[i, signal_col]
        
        # 1. Kiểm tra các lệnh đang mở có hit TP/SL không
        orders_to_remove = []
        for order_idx, order in enumerate(active_orders):
            order_closed = False
            
            if order['type'] == 'long':
                # Long position: SL < entry < TP
                if current_low <= order['sl']:
                    # Hit Stop Loss
                    losses += 1
                    loss_amount = order['amount'] * (sl_percent + 2 * fee_percent) * leverage / 100
                    balance -= loss_amount
                    order_closed = True
                elif current_high >= order['tp']:
                    # Hit Take Profit
                    wins += 1
                    buy_wins += 1
                    profit_amount = order['amount'] * (tp_percent - 2 * fee_percent) * leverage / 100
                    balance += profit_amount
                    order_closed = True
                    
            elif order['type'] == 'short':
                # Short position: TP < entry < SL
                if current_high >= order['sl']:
                    # Hit Stop Loss
                    losses += 1
                    loss_amount = order['amount'] * (sl_percent + 2 * fee_percent) * leverage / 100
                    balance -= loss_amount
                    order_closed = True
                elif current_low <= order['tp']:
                    # Hit Take Profit
                    wins += 1
                    sell_wins += 1
                    profit_amount = order['amount'] * (tp_percent - 2 * fee_percent) * leverage / 100
                    balance += profit_amount
                    order_closed = True
            
            if order_closed:
                orders_to_remove.append(order_idx)
        
        # Xóa các lệnh đã đóng (xóa từ cuối lên đầu để không bị lỗi index)
        for idx in reversed(orders_to_remove):
            active_orders.pop(idx)
        
        # 2. Kiểm tra tín hiệu mới và mở lệnh nếu có thể
        if signal in [0, 1] and len(active_orders) < max_orders and not account_blown:
            
            if signal == 1:  # Long position
                entry_price = current_close
                tp_price = entry_price * (1 + tp_percent / 100)
                sl_price = entry_price * (1 - sl_percent / 100)
                
                order = {
                    'type': 'long',
                    'entry_price': entry_price,
                    'tp': tp_price,
                    'sl': sl_price,
                    'amount': fixed_order_amount,  # Số tiền cố định
                    'entry_time': current_time
                }
                active_orders.append(order)
                total_signals += 1
                buy_signals += 1
                
            elif signal == 0:  # Short position
                entry_price = current_close
                tp_price = entry_price * (1 - tp_percent / 100)
                sl_price = entry_price * (1 + sl_percent / 100)
                
                order = {
                    'type': 'short',
                    'entry_price': entry_price,
                    'tp': tp_price,
                    'sl': sl_price,
                    'amount': fixed_order_amount,  # Số tiền cố định
                    'entry_time': current_time
                }
                active_orders.append(order)
                total_signals += 1
                sell_signals += 1
        
        # 3. Lưu lại lịch sử balance
        balance_history.append(balance)
        time_history.append(current_time)
    
    # Tính toán thống kê cuối cùng
    win_rate = (wins / total_signals * 100) if total_signals > 0 else 0.0
    buy_win_rate = (buy_wins / buy_signals * 100) if buy_signals > 0 else 0.0
    sell_win_rate = (sell_wins / sell_signals * 100) if sell_signals > 0 else 0.0
    total_return = (balance - initial_balance) / initial_balance * 100
    
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
        'time_history': time_history,
        'remaining_orders': len(active_orders),  # Số lệnh chưa đóng
        'max_concurrent_orders': max_orders,
        'position_size (%)': position_size * 100,
        'fixed_order_amount': fixed_order_amount,
        'account_blown': account_blown,
        'account_stop_loss': account_stop_loss
    }
    return result

# Đọc dữ liệu từ CSV
df = pd.read_csv('data.csv')

# Nếu không có Date thì tạo index datetime giả
if 'Date' not in df.columns:
    df['Date'] = pd.date_range(start='2022-01-01', periods=len(df), freq='H')

# Đánh giá với cài đặt mới
initial_balance = 1.0
result = evaluate_signals_realistic(
    df, 
    tp_percent=1, 
    sl_percent=1, 
    fee_percent=0.05, 
    leverage=5,
    max_orders=5,
    position_size=0.2,  # 20% vốn ban đầu mỗi lệnh
    initial_balance=initial_balance,
    account_stop_loss=0.5  # Dừng nếu tài khoản < 50% vốn ban đầu
)

# In thống kê chi tiết
print("=== KẾT QUẢ ĐÁNH GIÁ GIAO DỊCH THỰC TẾ ===")
print(f"Đòn bẩy: x{result['leverage']}")
print(f"Tối đa {result['max_concurrent_orders']} lệnh cùng lúc")
print(f"Mỗi lệnh: {result['position_size (%)']}% vốn ban đầu = {result['fixed_order_amount']}")
print(f"Account Stop Loss: {result['account_stop_loss'] * 100}% vốn ban đầu")
print()
if result['account_blown']:
    print("⚠️  TÀI KHOẢN BỊ THỔI (< 50% vốn ban đầu)")
    print()
print("THỐNG KÊ TÍN HIỆU:")
print(f"Tổng tín hiệu: {result['total_signals']}")
print(f"  - Tín hiệu mua (Long): {result['buy_signals']}")
print(f"  - Tín hiệu bán (Short): {result['sell_signals']}")
print()
print("THỐNG KÊ KẾT QUẢ:")
print(f"Win: {result['wins']}, Loss: {result['losses']}")
print(f"  - Buy wins: {result['buy_wins']}")
print(f"  - Sell wins: {result['sell_wins']}")
print(f"Tỉ lệ thắng tổng: {result['win_rate']}%")
print(f"  - Tỉ lệ thắng Buy: {result['buy_win_rate']}%")
print(f"  - Tỉ lệ thắng Sell: {result['sell_win_rate']}%")
print()
print("KẾT QUẢ TÀI CHÍNH:")
print(f"Số vốn ban đầu: {initial_balance}")
print(f"Số vốn cuối cùng: {result['final_balance']}")
print(f"Tổng lợi nhuận: {result['total_return (%)']}%")
print(f"Số lệnh chưa đóng: {result['remaining_orders']}")