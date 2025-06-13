import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
from scipy.signal import argrelextrema
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# FastAPI app
app = FastAPI(title="Crypto Chart Pattern Analyzer API", version="1.0.0")

# Load biến môi trường
load_dotenv()
password = os.getenv('MONGO_PASSWORD')

# Kết nối MongoDB Atlas
uri = f"mongodb+srv://trainguyenchi30:{password}@cryptodata.t2i1je2.mongodb.net/?retryWrites=true&w=majority&appName=CryptoData"
client = MongoClient(uri, server_api=ServerApi('1'))

# Truy cập DB và Collection
db = client['my_database']
collection = db['AI_prediction']

# Kết nối với Binance
binance = ccxt.binance({
    'apiKey': '',
    'secret': '',
})


class RealtimeCryptoPatternAnalyzer:
    def __init__(self):
        self.df = None
        
        # Đỉnh/đáy order=1 (realtime)
        self.realtime_highs = None
        self.realtime_lows = None
        
        # Đỉnh/đáy order=5 (đáng tin cậy)
        self.reliable_highs = None
        self.reliable_lows = None
    
    def fetch_data(self, symbol='BTC/USDT', timeframe='1h', count=1000):
        """Lấy dữ liệu OHLCV từ Binance"""
        try:
            ohlcv = binance.fetch_ohlcv(symbol, timeframe=timeframe, limit=count)
            self.df = pd.DataFrame(ohlcv, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
            self.df['Date'] = pd.to_datetime(self.df['Date'], unit='ms')
            self.df['signal'] = -1  # Mặc định signal là -1
            return self.df
        except Exception as e:
            raise Exception(f"Error fetching data: {e}")
    
    def find_peaks_and_troughs(self, order_realtime=1, order_reliable=20):
        """
        Tìm đỉnh/đáy với 2 mức độ tin cậy:
        - order_realtime: Cho trading realtime (thường là 1)
        - order_reliable: Cho các đỉnh/đáy đáng tin cậy (thường là 5)
        """
        if self.df is None or len(self.df) == 0:
            raise ValueError("Cần fetch dữ liệu trước")
            
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
                
                # Kiểm tra điều kiện Double Top
                price_diff = abs(current_high_price - reliable_high_price) / reliable_high_price
                
                if price_diff <= threshold:
                    # Kiểm tra có đáy ở giữa không (để xác nhận là double top)
                    valley_between = self.find_valley_between(reliable_high_idx, current_high_idx)
                    
                    if valley_between is not None:
                        # Tín hiệu bán xuất hiện ngay sau đỉnh hiện tại
                        if current_high_idx + 1 < len(self.df):
                            signals[current_high_idx + 1] = 0  # Tín hiệu bán
                        break
        
        return signals
    
    def detect_realtime_double_bottom(self, threshold=0.02, lookback_window=50):
        """
        Phát hiện Double Bottom với logic thực tế:
        - Sử dụng đáy realtime (order=1) hiện tại
        - So sánh với đáy reliable (order=5) trước đó trong khoảng lookback_window
        """
        if self.realtime_lows is None or self.reliable_lows is None:
            raise ValueError("Cần chạy find_peaks_and_troughs() trước")
        
        signals = np.full(len(self.df), -1)
        
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
                
                # Kiểm tra điều kiện Double Bottom
                price_diff = abs(current_low_price - reliable_low_price) / reliable_low_price
                
                if price_diff <= threshold:
                    # Kiểm tra có đỉnh ở giữa không (để xác nhận là double bottom)
                    peak_between = self.find_peak_between(reliable_low_idx, current_low_idx)
                    
                    if peak_between is not None:
                        # Tín hiệu mua xuất hiện ngay sau đáy hiện tại
                        if current_low_idx + 1 < len(self.df):
                            signals[current_low_idx + 1] = 1  # Tín hiệu mua
                        break
        
        return signals
    
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
    
    def detect_realtime_head_and_shoulders(self, threshold=0.02, lookback_window=100):
        """
        Phát hiện Head and Shoulders với logic thực tế:
        - Sử dụng đỉnh realtime hiện tại làm right shoulder
        - Tìm head và left shoulder từ các đỉnh reliable trước đó
        """
        if self.realtime_highs is None or self.reliable_highs is None:
            raise ValueError("Cần chạy find_peaks_and_troughs() trước")
        
        signals = np.full(len(self.df), -1)
        
        # Duyệt qua các đỉnh realtime làm right shoulder
        for right_shoulder_idx in self.realtime_highs:
            right_shoulder_price = self.df['High'].iloc[right_shoulder_idx]
            
            # Tìm các đỉnh reliable trước đó để làm head và left shoulder
            reliable_highs_before = [
                idx for idx in self.reliable_highs 
                if right_shoulder_idx - lookback_window <= idx < right_shoulder_idx
            ]
            
            if len(reliable_highs_before) < 2:
                continue
            
            # Thử các kết hợp head và left shoulder
            for i in range(len(reliable_highs_before) - 1):
                for j in range(i + 1, len(reliable_highs_before)):
                    left_shoulder_idx = reliable_highs_before[i]
                    head_idx = reliable_highs_before[j]
                    
                    left_shoulder_price = self.df['High'].iloc[left_shoulder_idx]
                    head_price = self.df['High'].iloc[head_idx]
                    
                    # Kiểm tra điều kiện H&S
                    if (head_price > left_shoulder_price and 
                        head_price > right_shoulder_price and
                        abs(left_shoulder_price - right_shoulder_price) / head_price <= threshold):
                        
                        # Tìm neckline
                        neck_left = self.find_valley_between(left_shoulder_idx, head_idx)
                        neck_right = self.find_valley_between(head_idx, right_shoulder_idx)
                        
                        if neck_left is not None and neck_right is not None:
                            # Tín hiệu bán xuất hiện ngay sau right shoulder
                            if right_shoulder_idx + 1 < len(self.df):
                                signals[right_shoulder_idx + 1] = 0  # Tín hiệu bán
                            break
                
                if signals[right_shoulder_idx + 1] == 0:  # Đã tìm thấy pattern
                    break
        
        return signals
    
    def detect_realtime_inverted_head_and_shoulders(self, threshold=0.02, lookback_window=100):
        """
        Phát hiện Inverted Head and Shoulders với logic thực tế:
        - Sử dụng đáy realtime hiện tại làm right shoulder
        - Tìm head và left shoulder từ các đáy reliable trước đó
        """
        if self.realtime_lows is None or self.reliable_lows is None:
            raise ValueError("Cần chạy find_peaks_and_troughs() trước")
        
        signals = np.full(len(self.df), -1)
        
        # Duyệt qua các đáy realtime làm right shoulder
        for right_shoulder_idx in self.realtime_lows:
            right_shoulder_price = self.df['Low'].iloc[right_shoulder_idx]
            
            # Tìm các đáy reliable trước đó để làm head và left shoulder
            reliable_lows_before = [
                idx for idx in self.reliable_lows 
                if right_shoulder_idx - lookback_window <= idx < right_shoulder_idx
            ]
            
            if len(reliable_lows_before) < 2:
                continue
            
            # Thử các kết hợp head và left shoulder
            for i in range(len(reliable_lows_before) - 1):
                for j in range(i + 1, len(reliable_lows_before)):
                    left_shoulder_idx = reliable_lows_before[i]
                    head_idx = reliable_lows_before[j]
                    
                    left_shoulder_price = self.df['Low'].iloc[left_shoulder_idx]
                    head_price = self.df['Low'].iloc[head_idx]
                    
                    # Kiểm tra điều kiện Inverted H&S
                    if (head_price < left_shoulder_price and 
                        head_price < right_shoulder_price and
                        abs(left_shoulder_price - right_shoulder_price) / head_price <= threshold):
                        
                        # Tìm neckline
                        neck_left = self.find_peak_between(left_shoulder_idx, head_idx)
                        neck_right = self.find_peak_between(head_idx, right_shoulder_idx)
                        
                        if neck_left is not None and neck_right is not None:
                            # Tín hiệu mua xuất hiện ngay sau right shoulder
                            if right_shoulder_idx + 1 < len(self.df):
                                signals[right_shoulder_idx + 1] = 1  # Tín hiệu mua
                            break
                
                if signals[right_shoulder_idx + 1] == 1:  # Đã tìm thấy pattern
                    break
        
        return signals
    
    def combine_signals(self, *signal_arrays):
        """Kết hợp nhiều mảng tín hiệu"""
        for signals in signal_arrays:
            self.df['signal'] = np.where(signals != -1, signals, self.df['signal'])
    
    def save_to_mongodb(self, symbol='BTC/USDT', timeframe='1h'):
        """Lưu kết quả phân tích vào MongoDB"""
        try:
            # Chuẩn bị dữ liệu để lưu
            data_to_save = []
            
            for index, row in self.df.iterrows():
                record = {
                    'Date': row['Date'],
                    'Open': float(row['Open']),
                    'High': float(row['High']),
                    'Low': float(row['Low']),
                    'Close': float(row['Close']),
                    'confidence': int(row['signal']),
                    'symbol': symbol
                }
                
                data_to_save.append(record)
            
            # Lưu vào MongoDB
            if data_to_save:
                # Xóa dữ liệu cũ
                collection.delete_many({'symbol': symbol})

                # Chèn dữ liệu mới
                result = collection.insert_many(data_to_save)
                return len(result.inserted_ids)
            else:
                return 0
                
        except Exception as e:
            raise Exception(f"Error saving to MongoDB: {e}")
    
    def run_analysis( self, symbol='BTC/USDT', timeframe='1h', count=1000, order_realtime=1, order_reliable=20, threshold=0.01, lookback_window=10):
        """Chạy phân tích và lưu vào MongoDB"""

        # Bước 1: Crawl dữ liệu
        self.fetch_data(symbol, timeframe, count)

        # Bước 2: Tìm đỉnh và đáy
        self.find_peaks_and_troughs(order_realtime=order_realtime, order_reliable=order_reliable)

        # Bước 3: Nhận diện các mô hình
        dt_signals = self.detect_realtime_double_top(threshold=threshold, lookback_window=lookback_window)
        db_signals = self.detect_realtime_double_bottom(threshold=threshold, lookback_window=lookback_window)
        hs_signals = self.detect_realtime_head_and_shoulders(threshold=threshold, lookback_window=lookback_window)
        ihs_signals = self.detect_realtime_inverted_head_and_shoulders(threshold=threshold, lookback_window=lookback_window)

        # Bước 4: Kết hợp tín hiệu
        self.combine_signals(dt_signals, db_signals, hs_signals, ihs_signals)

        # Bước 5: Lưu vào MongoDB
        return self.save_to_mongodb(symbol, timeframe)



# ===============================
# FASTAPI ENDPOINTS
# ===============================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Crypto Chart Pattern Analyzer API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/ping")
async def ping():
    """
    Endpoint để cập nhật dữ liệu mới nhất vào MongoDB
    """
    try:
        # Tạo analyzer
        analyzer = RealtimeCryptoPatternAnalyzer()
        
        # Chạy phân tích và lưu dữ liệu
        analyzer.run_analysis(symbol='BTC/USDT', timeframe='1h', order_realtime=1, order_reliable=20, threshold=0.01, lookback_window=10)
        analyzer.run_analysis(symbol='ETH/USDT', timeframe='1h', order_realtime=1, order_reliable=20, threshold=0.03, lookback_window=10)
        analyzer.run_analysis(symbol='SOL/USDT', timeframe='1h', order_realtime=1, order_reliable=20, threshold=0.02, lookback_window=10)
        analyzer.run_analysis(symbol='XRP/USDT', timeframe='1h', order_realtime=1, order_reliable=20, threshold=0.015, lookback_window=10)
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "DONE"
            }
        )
            
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            content={
                "message": "ERROR"
            }
        )


def start_api_server(host="0.0.0.0", port=8000):
    """Khởi động FastAPI server"""
    print(f"🚀 Starting FastAPI server on {host}:{port}")
    print(f"📖 API Documentation: http://{host}:{port}/docs")
    print(f"🔍 Endpoint: GET /ping - Update latest data to MongoDB")
    print("-" * 50)
    
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_api_server()