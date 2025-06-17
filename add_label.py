import pandas as pd
import numpy as np

def analyze_signal_quality(csv_file='data.csv', window_size=12, min_change=0.5, ratio_threshold=3):
    """
    Phân tích chất lượng tín hiệu từ file CSV
    
    Parameters:
    csv_file (str): Đường dẫn file CSV
    window_size (int): Kích thước cửa sổ để tính future_high/low
    min_change (float): % thay đổi tối thiểu để coi là tín hiệu mạnh
    ratio_threshold (float): Tỉ lệ gain/loss tối thiểu để coi là tín hiệu rõ ràng
    """
    # Load dữ liệu
    data = pd.read_csv(csv_file, parse_dates=['Date'])
    print(f"Đã load {len(data)} dòng dữ liệu từ {csv_file}")
    
    # Tính future_high và future_low giống như công thức đã cho
    data['future_high'] = data['High'].shift(-1).rolling(window=window_size, min_periods=window_size).max().shift(-window_size + 1)
    data['future_low'] = data['Low'].shift(-1).rolling(window=window_size, min_periods=window_size).min().shift(-window_size + 1)
    
    # Tính % thay đổi giá
    data['max_gain_pct'] = ((data['future_high'] - data['Close']) / data['Close'] * 100).round(2)
    data['max_loss_pct'] = ((data['Close'] - data['future_low']) / data['Close'] * 100).round(2)
    
    # Hàm gán nhãn - Chỉ tính toán khi signal khác -1
    def label_comprehensive(row):
        # Chỉ tính quality_label khi có signal thực sự (signal != -1)
        if row['signal'] == -1:
            return None  # Không có tín hiệu, không cần đánh giá
        
        # Tính toán quality_label cho các điểm có tín hiệu
        if (row['max_gain_pct'] >= min_change and 
            row['max_gain_pct'] / max(row['max_loss_pct'], 0.1) >= ratio_threshold):
            return 1  # Tăng rõ ràng
        elif (row['max_loss_pct'] >= min_change and 
              row['max_loss_pct'] / max(row['max_gain_pct'], 0.1) >= ratio_threshold):
            return 0  # Giảm rõ ràng
        else:
            return -1  # Không rõ ràng
    
    # Áp dụng hàm gán nhãn
    data['quality_label'] = data.apply(label_comprehensive, axis=1)
    
    # Phân tích tín hiệu
    def analyze_signals():
        # Lọc các điểm có tín hiệu
        buy_signals = data[data['signal'] == 1].copy()
        sell_signals = data[data['signal'] == 0].copy()
        
        print("\n" + "="*60)
        print("PHÂN TÍCH CHẤT LƯỢNG TÍN HIỆU")
        print(f"Tham số: min_change={min_change}%, ratio_threshold={ratio_threshold}")
        print("="*60)
        
        # Phân tích tín hiệu mua (signal = 1)
        if len(buy_signals) > 0:
            print(f"\n📈 TÍN HIỆU MUA (Signal = 1): {len(buy_signals)} điểm")
            
            # Đếm các loại kết quả
            strong_up = len(buy_signals[buy_signals['quality_label'] == 1])
            strong_down = len(buy_signals[buy_signals['quality_label'] == 0])
            unclear = len(buy_signals[buy_signals['quality_label'] == -1])
            valid_signals = len(buy_signals[buy_signals['quality_label'].notna()])
            
            print(f"  ✅ Tăng mạnh thực sự: {strong_up}/{valid_signals} ({strong_up/valid_signals*100:.1f}%)")
            print(f"  ❌ Giảm mạnh (sai): {strong_down}/{valid_signals} ({strong_down/valid_signals*100:.1f}%)")
            print(f"  ⚪ Không rõ ràng: {unclear}/{valid_signals} ({unclear/valid_signals*100:.1f}%)")
            
            # Thống kê chi tiết
            avg_gain = buy_signals['max_gain_pct'].mean()
            avg_loss = buy_signals['max_loss_pct'].mean()
            print(f"  📊 Avg Max Gain: {avg_gain:.2f}%")
            print(f"  📊 Avg Max Loss: {avg_loss:.2f}%")
            
        # Phân tích tín hiệu bán (signal = 0)
        if len(sell_signals) > 0:
            print(f"\n📉 TÍN HIỆU BÁN (Signal = 0): {len(sell_signals)} điểm")
            
            # Đếm các loại kết quả
            strong_down = len(sell_signals[sell_signals['quality_label'] == 0])
            strong_up = len(sell_signals[sell_signals['quality_label'] == 1])
            unclear = len(sell_signals[sell_signals['quality_label'] == -1])
            valid_signals = len(sell_signals[sell_signals['quality_label'].notna()])
            
            print(f"  ✅ Giảm mạnh thực sự: {strong_down}/{valid_signals} ({strong_down/valid_signals*100:.1f}%)")
            print(f"  ❌ Tăng mạnh (sai): {strong_up}/{valid_signals} ({strong_up/valid_signals*100:.1f}%)")
            print(f"  ⚪ Không rõ ràng: {unclear}/{valid_signals} ({unclear/valid_signals*100:.1f}%)")
            
            # Thống kê chi tiết
            avg_gain = sell_signals['max_gain_pct'].mean()
            avg_loss = sell_signals['max_loss_pct'].mean()
            print(f"  📊 Avg Max Gain: {avg_gain:.2f}%")
            print(f"  📊 Avg Max Loss: {avg_loss:.2f}%")
        
        return buy_signals, sell_signals
    
    # Chạy phân tích
    buy_signals, sell_signals = analyze_signals()
    
    # Tạo bảng tóm tắt
    def create_summary_table():
        print(f"\n" + "="*60)
        print("BẢNG TÓM TẮT CHẤT LƯỢNG TÍN HIỆU")
        print("="*60)
        
        summary_data = []
        
        # Tín hiệu mua
        if len(buy_signals) > 0:
            valid_buy = buy_signals[buy_signals['quality_label'].notna()]
            buy_accuracy = len(valid_buy[valid_buy['quality_label'] == 1]) / len(valid_buy) * 100 if len(valid_buy) > 0 else 0
            summary_data.append({
                'Signal Type': 'BUY (Signal=1)',
                'Total Signals': len(buy_signals),
                'Strong Move Correct': len(buy_signals[buy_signals['quality_label'] == 1]),
                'Strong Move Wrong': len(buy_signals[buy_signals['quality_label'] == 0]),
                'Unclear': len(buy_signals[buy_signals['quality_label'] == -1]),
                'Accuracy (%)': f"{buy_accuracy:.1f}%"
            })
        
        # Tín hiệu bán
        if len(sell_signals) > 0:
            valid_sell = sell_signals[sell_signals['quality_label'].notna()]
            sell_accuracy = len(valid_sell[valid_sell['quality_label'] == 0]) / len(valid_sell) * 100 if len(valid_sell) > 0 else 0
            summary_data.append({
                'Signal Type': 'SELL (Signal=0)',
                'Total Signals': len(sell_signals),
                'Strong Move Correct': len(sell_signals[sell_signals['quality_label'] == 0]),
                'Strong Move Wrong': len(sell_signals[sell_signals['quality_label'] == 1]),
                'Unclear': len(sell_signals[sell_signals['quality_label'] == -1]),
                'Accuracy (%)': f"{sell_accuracy:.1f}%"
            })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            print(summary_df.to_string(index=False))
    
    create_summary_table()
    
    # Lưu kết quả
    output_file = 'data.csv'
    data.to_csv(output_file, index=False)
    print(f"\n💾 Đã lưu kết quả phân tích vào file: {output_file}")
    
    # Trả về DataFrame để sử dụng tiếp
    return data, buy_signals, sell_signals

# Ví dụ sử dụng với các tham số khác nhau
if __name__ == "__main__":
    # Sử dụng tham số mặc định
    print("PHÂN TÍCH VỚI THAM SỐ MẶC ĐỊNH:")
    data1, buy1, sell1 = analyze_signal_quality('data.csv', window_size=20, min_change=1, ratio_threshold=1)
    