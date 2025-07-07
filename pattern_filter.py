import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt


class RealtimeChartPatternAnalyzer:
    def __init__(self, csv_file='OHLCV.csv'):
        self.df = pd.read_csv(csv_file, parse_dates=['Date'])
        self.df['signal'] = -1  # Mặc định signal là -1
        self.df['price_distance'] = np.nan  # Thêm cột lưu khoảng cách giá
        
        # Đỉnh/đáy order=1 (realtime)
        self.realtime_highs = None
        self.realtime_lows = None
        
        # Đỉnh/đáy order=5 (đáng tin cậy)
        self.reliable_highs = None
        self.reliable_lows = None
    
    def find_peaks_and_troughs(self, order_realtime=1, order_reliable=20):
        """
        Tìm đỉnh/đáy với 2 mức độ tin cậy:
        - order_realtime: Cho trading realtime (thường là 1)
        - order_reliable: Cho các đỉnh/đáy đáng tin cậy (thường là 5)
        """
        # Đỉnh/đáy realtime (order=1)
        self.realtime_highs = argrelextrema(self.df['Close'].values, np.greater, order=order_realtime)[0]
        self.realtime_lows = argrelextrema(self.df['Close'].values, np.less, order=order_realtime)[0]
        
        # Đỉnh/đáy đáng tin cậy (order=5)
        self.reliable_highs = argrelextrema(self.df['Close'].values, np.greater, order=order_reliable)[0]
        self.reliable_lows = argrelextrema(self.df['Close'].values, np.less, order=order_reliable)[0]
        
        return {
            'realtime_highs': self.realtime_highs,
            'realtime_lows': self.realtime_lows,
            'reliable_highs': self.reliable_highs,
            'reliable_lows': self.reliable_lows
        }
    
    def detect_realtime_double_top(self, threshold=0.02, lookback_window=50):
        """
        Phát hiện Double Top với logic thực tế:
        - Sử dụng đỉnh realtime (order=1) hiện tại
        - So sánh với đỉnh reliable (order=5) trước đó trong khoảng lookback_window
        """
        if self.realtime_highs is None or self.reliable_highs is None:
            raise ValueError("Cần chạy find_peaks_and_troughs() trước")
        
        signals = np.full(len(self.df), -1)
        price_distances = np.full(len(self.df), np.nan)
        
        # Duyệt qua các đỉnh realtime
        for current_high_idx in self.realtime_highs:
            current_high_price = self.df['High'].iloc[current_high_idx]
            
            # Tìm đỉnh reliable trước đó trong khoảng lookback_window
            reliable_highs_before = [
                idx for idx in self.reliable_highs 
                if current_high_idx - lookback_window <= idx < current_high_idx
            ]
            
            if not reliable_highs_before:
                continue
            
            # Tìm đỉnh reliable gần nhất và cao nhất
            for reliable_high_idx in reliable_highs_before:
                reliable_high_price = self.df['High'].iloc[reliable_high_idx]
                
                # Tính khoảng cách giá
                price_diff = abs(current_high_price - reliable_high_price) / reliable_high_price
                
                # Kiểm tra điều kiện Double Top
                if price_diff <= threshold:
                    # Kiểm tra có đáy ở giữa không (để xác nhận là double top)
                    valley_between = self.find_valley_between(reliable_high_idx, current_high_idx)
                    
                    if valley_between is not None:
                        # Tín hiệu bán xuất hiện ngay sau đỉnh hiện tại
                        if current_high_idx + 1 < len(self.df):
                            signals[current_high_idx + 1] = 0  # Tín hiệu bán
                            price_distances[current_high_idx + 1] = price_diff  # Lưu khoảng cách
                        break
        
        return signals, price_distances
    
    def detect_realtime_double_bottom(self, threshold=0.02, lookback_window=50):
        """
        Phát hiện Double Bottom với logic thực tế:
        - Sử dụng đáy realtime (order=1) hiện tại
        - So sánh với đáy reliable (order=5) trước đó trong khoảng lookback_window
        """
        if self.realtime_lows is None or self.reliable_lows is None:
            raise ValueError("Cần chạy find_peaks_and_troughs() trước")
        
        signals = np.full(len(self.df), -1)
        price_distances = np.full(len(self.df), np.nan)
        
        # Duyệt qua các đáy realtime
        for current_low_idx in self.realtime_lows:
            current_low_price = self.df['Low'].iloc[current_low_idx]
            
            # Tìm đáy reliable trước đó trong khoảng lookback_window
            reliable_lows_before = [
                idx for idx in self.reliable_lows 
                if current_low_idx - lookback_window <= idx < current_low_idx
            ]
            
            if not reliable_lows_before:
                continue
            
            # Tìm đáy reliable gần nhất và thấp nhất
            for reliable_low_idx in reliable_lows_before:
                reliable_low_price = self.df['Low'].iloc[reliable_low_idx]
                
                # Tính khoảng cách giá
                price_diff = abs(current_low_price - reliable_low_price) / reliable_low_price
                
                # Kiểm tra điều kiện Double Bottom
                if price_diff <= threshold:
                    # Kiểm tra có đỉnh ở giữa không (để xác nhận là double bottom)
                    peak_between = self.find_peak_between(reliable_low_idx, current_low_idx)
                    
                    if peak_between is not None:
                        # Tín hiệu mua xuất hiện ngay sau đáy hiện tại
                        if current_low_idx + 1 < len(self.df):
                            signals[current_low_idx + 1] = 1  # Tín hiệu mua
                            price_distances[current_low_idx + 1] = price_diff  # Lưu khoảng cách
                        break
        
        return signals, price_distances
    
    def find_valley_between(self, start_idx, end_idx):
        """Tìm đáy thấp nhất giữa hai điểm"""
        if start_idx >= end_idx:
            return None
        
        valley_section = self.df['Low'].iloc[start_idx:end_idx+1]
        if len(valley_section) == 0:
            return None
        
        min_idx = valley_section.idxmin()
        return min_idx
    
    def find_peak_between(self, start_idx, end_idx):
        """Tìm đỉnh cao nhất giữa hai điểm"""
        if start_idx >= end_idx:
            return None
        
        peak_section = self.df['High'].iloc[start_idx:end_idx+1]
        if len(peak_section) == 0:
            return None
        
        max_idx = peak_section.idxmax()
        return max_idx
    
    def combine_signals(self, signal_arrays_with_distances):
        """Kết hợp nhiều mảng tín hiệu và khoảng cách"""
        for signals, distances in signal_arrays_with_distances:
            # Cập nhật signal
            self.df['signal'] = np.where(signals != -1, signals, self.df['signal'])
            # Cập nhật price_distance
            self.df['price_distance'] = np.where(~np.isnan(distances), distances, self.df['price_distance'])
    
    def validate_predictions(self, future_window=12):
        """Kiểm tra độ chính xác của dự đoán"""
        self.df['future_close'] = self.df['Close'].shift(-future_window)
        self.df['actual_change'] = self.df['future_close'] / self.df['Close'] * 100 - 100 
        self.df['prediction_correct'] = -1
        
        # Kiểm tra tín hiệu mua (signal = 1)
        self.df.loc[self.df['signal'] == 1, 'prediction_correct'] = (
            self.df['actual_change'] > 0
        ).astype(int)
        
        # Kiểm tra tín hiệu bán (signal = 0)
        self.df.loc[self.df['signal'] == 0, 'prediction_correct'] = (
            self.df['actual_change'] < 0
        ).astype(int)
    
    def calculate_accuracy_for_pattern(self, pattern_signals, signal_value):
        """Tính độ chính xác cho một mô hình cụ thể"""
        pattern_df = self.df[(pattern_signals == signal_value) & (self.df['future_close'].notna())]
        if len(pattern_df) == 0:
            return 0, 0
        
        correct = pattern_df['prediction_correct'].sum()
        total = len(pattern_df)
        accuracy = correct / total if total > 0 else 0
        return accuracy, total
    
    def get_accuracy_report(self, dt_signals, db_signals):
        """Tạo báo cáo độ chính xác cho các pattern"""
        dt_accuracy, dt_count = self.calculate_accuracy_for_pattern(dt_signals, 0)
        db_accuracy, db_count = self.calculate_accuracy_for_pattern(db_signals, 1)
        
        return {
            'double_top': {'accuracy': dt_accuracy, 'count': dt_count},
            'double_bottom': {'accuracy': db_accuracy, 'count': db_count}
        }
    
    def plot_realtime_analysis(self, num_sessions=500):
        """Vẽ biểu đồ phân tích realtime chỉ gồm biểu đồ giá và tín hiệu"""
        df_plot = self.df.iloc[-num_sessions:].copy()

        plt.figure(figsize=(16, 6))  # Chiều cao nhỏ lại vì chỉ còn 1 biểu đồ

        # Biểu đồ giá và tín hiệu
        plt.plot(df_plot['Date'], df_plot['Close'], label='Close Price', color='black', linewidth=1)

        reliable_highs_plot = df_plot.index[df_plot.index.isin(self.reliable_highs)]
        reliable_lows_plot = df_plot.index[df_plot.index.isin(self.reliable_lows)]

        if len(reliable_highs_plot) > 0:
            plt.scatter(df_plot.loc[reliable_highs_plot, 'Date'], 
                    df_plot.loc[reliable_highs_plot, 'High'], 
                    color='orange', s=30, label='Reliable Highs', zorder=4)

        if len(reliable_lows_plot) > 0:
            plt.scatter(df_plot.loc[reliable_lows_plot, 'Date'], 
                    df_plot.loc[reliable_lows_plot, 'Low'], 
                    color='purple', s=30, label='Reliable Lows', zorder=4)

        correct_buy = df_plot[(df_plot['signal'] == 1) & (df_plot['prediction_correct'] == 1)]
        wrong_buy = df_plot[(df_plot['signal'] == 1) & (df_plot['prediction_correct'] == 0)]
        correct_sell = df_plot[(df_plot['signal'] == 0) & (df_plot['prediction_correct'] == 1)]
        wrong_sell = df_plot[(df_plot['signal'] == 0) & (df_plot['prediction_correct'] == 0)]

        plt.scatter(correct_buy['Date'], correct_buy['Close'], 
                color='green', label='Buy Correct', s=100, marker='^', zorder=5)
        plt.scatter(correct_sell['Date'], correct_sell['Close'], 
                color='red', label='Sell Correct', s=100, marker='v', zorder=5)
        plt.scatter(wrong_buy['Date'], wrong_buy['Close'], 
                color='lightgreen', label='Buy Wrong', s=100, marker='^', zorder=5)
        plt.scatter(wrong_sell['Date'], wrong_sell['Close'], 
                color='pink', label='Sell Wrong', s=100, marker='v', zorder=5)

        plt.title(f'Realtime Chart Pattern Analysis (Last {num_sessions} Sessions)')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    
    def run_realtime_analysis(self, order_realtime=1, order_reliable=20, 
                             threshold=0.01, lookback_window=10, future_window=10):
        """Chạy phân tích realtime đầy đủ"""
        
        # Bước 1: Tìm đỉnh và đáy
        peaks_troughs = self.find_peaks_and_troughs(order_realtime, order_reliable)
        
        print(f"Tìm thấy:")
        print(f"  - {len(self.realtime_highs)} đỉnh realtime (order={order_realtime})")
        print(f"  - {len(self.realtime_lows)} đáy realtime (order={order_realtime})")
        print(f"  - {len(self.reliable_highs)} đỉnh reliable (order={order_reliable})")
        print(f"  - {len(self.reliable_lows)} đáy reliable (order={order_reliable})")
        
        # Bước 2: Nhận diện các mô hình
        dt_signals, dt_distances = self.detect_realtime_double_top(threshold, lookback_window)
        db_signals, db_distances = self.detect_realtime_double_bottom(threshold, lookback_window)
        
        # Bước 3: Kết hợp tín hiệu
        self.combine_signals([(dt_signals, dt_distances), (db_signals, db_distances)])
        
        # Bước 4: Kiểm tra độ chính xác
        self.validate_predictions(future_window)
        
        # Bước 5: Tạo báo cáo
        report = self.get_accuracy_report(dt_signals, db_signals)
        
        return report
    
    def save_results(self, filename='data.csv'):
        """Lưu kết quả phân tích"""
        self.df.to_csv(filename, index=False)
        print(f"Đã lưu kết quả vào file '{filename}'")


def print_realtime_analysis_report(report):
    """In báo cáo phân tích realtime"""
    print("\n" + "="*60)
    print("KẾT QUẢ PHÂN TÍCH REALTIME CHART PATTERNS")
    print("="*60)
    
    print(f"\nDouble Top (Tín hiệu bán):")
    print(f"  - Số lần phát hiện: {report['double_top']['count']}")
    print(f"  - Độ chính xác: {report['double_top']['accuracy']:.2%}")
    
    print(f"\nDouble Bottom (Tín hiệu mua):")
    print(f"  - Số lần phát hiện: {report['double_bottom']['count']}")
    print(f"  - Độ chính xác: {report['double_bottom']['accuracy']:.2%}")
    
    # Tính tổng kết
    total_signals = sum([report[pattern]['count'] for pattern in report])
    if total_signals > 0:
        total_correct = sum([
            report[pattern]['count'] * report[pattern]['accuracy'] 
            for pattern in report
        ])
        overall_accuracy = total_correct / total_signals
        print(f"\nTỔNG KẾT:")
        print(f"  - Tổng số tín hiệu: {total_signals}")
        print(f"  - Độ chính xác tổng thể: {overall_accuracy:.2%}")
    
    print("="*60)


def main():
    """Hàm main để chạy phân tích"""
    # Khởi tạo analyzer
    analyzer = RealtimeChartPatternAnalyzer('OHLCV.csv')
    
    # Chạy phân tích realtime với các tham số
    report = analyzer.run_realtime_analysis(
        order_realtime=1,      # Đỉnh/đáy realtime
        order_reliable=20,     # Đỉnh/đáy đáng tin cậy
        threshold=0.02,        # Ngưỡng chênh lệch giá 2%
        lookback_window=10,    # Tìm kiếm trong 50 nến trước
        future_window=10       # Đánh giá kết quả sau 10 nến
    )
    
    # In báo cáo
    print_realtime_analysis_report(report)
    
    # Lưu kết quả
    analyzer.save_results()
    
    # Vẽ biểu đồ
    analyzer.plot_realtime_analysis(num_sessions=200)

if __name__ == "__main__":
    main()