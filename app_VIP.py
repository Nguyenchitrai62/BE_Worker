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
import pickle
import tensorflow as tf

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


class AIModelPredictor:
    """Class để load và sử dụng 2 AI models: long và short"""
    
    def __init__(self, model_configs=None):
        """
        model_configs: dict với format:
        {
            'long': 'models/best_long_model.h5',
            'short': 'models/best_short_model.h5'
        }
        """
        if model_configs is None:
            model_configs = {
                'long': 'models/best_long_model.h5',
                'short': 'models/best_short_model.h5'
            }
        
        self.model_configs = model_configs
        self.models = {}
        self.scalers_X = {}
        self.scalers_y = {}
        self.model_infos = {}
        self.load_all_models()
    
    def load_model_by_type(self, model_type, model_path):
        """Load một model cụ thể theo type (long hoặc short)"""
        try:
            print(f"📥 Loading {model_type} model from {model_path}...")
            
            # Load model
            model = tf.keras.models.load_model(model_path)
            
            # Load scalers và model info
            scaler_X_path = f'models/{model_type}_scaler_X.pkl'
            scaler_y_path = f'models/{model_type}_scaler_y.pkl'
            model_info_path = f'models/{model_type}_model_info.pkl'
            
            with open(scaler_X_path, 'rb') as f:
                scaler_X = pickle.load(f)
            
            with open(scaler_y_path, 'rb') as f:
                scaler_y = pickle.load(f)
            
            with open(model_info_path, 'rb') as f:
                model_info = pickle.load(f)
            
            # Lưu vào dictionaries
            self.models[model_type] = model
            self.scalers_X[model_type] = scaler_X
            self.scalers_y[model_type] = scaler_y
            self.model_infos[model_type] = model_info
            
            print(f"✅ {model_type.capitalize()} model loaded successfully!")
            print(f"📊 {model_type} info: R² = {model_info['r2_score']:.4f}, RMSE = {model_info['rmse']:.6f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading {model_type} model: {e}")
            return False
    
    def load_all_models(self):
        """Load cả 2 models: long và short"""
        success_count = 0
        for model_type, model_path in self.model_configs.items():
            if self.load_model_by_type(model_type, model_path):
                success_count += 1
        
        print(f"🎯 Successfully loaded {success_count}/{len(self.model_configs)} models")
    
    def prepare_features(self, df):
        """Chuẩn bị features giống như trong training"""
        df_copy = df.copy()
        
        # Tính các chỉ báo (giống như trong training)
        df_copy['ma7'] = df_copy['Close'].rolling(window=7).mean()
        df_copy['price_vs_ma7'] = (df_copy['Close'] - df_copy['ma7']) / df_copy['ma7']
        df_copy['close/open'] = df_copy['Close'] / df_copy['Open'] - 1
        
        # Loại bỏ NaN
        df_copy = df_copy.dropna(subset=['price_vs_ma7', 'close/open'])
        
        return df_copy
    
    def predict_max_gain_pct_single(self, sequence, features, model_type='long'):
        """Dự đoán max_gain_pct cho một sequence với model cụ thể"""
        if model_type not in self.models:
            print(f"❌ Model {model_type} not available")
            return None
        
        try:
            model = self.models[model_type]
            scaler_X = self.scalers_X[model_type]
            scaler_y = self.scalers_y[model_type]
            
            # Chuẩn hóa sequence
            seq_scaled = scaler_X.transform(sequence.reshape(-1, len(features)))
            seq_scaled = seq_scaled.reshape(1, len(sequence), len(features))
            
            # Predict scaled value
            pred_scaled = model.predict(seq_scaled, verbose=0)[0][0]
            
            # Convert back to original scale
            pred_original = scaler_y.inverse_transform([[pred_scaled]])[0][0]
            
            return pred_original
        except Exception as e:
            print(f"⚠️ Error in {model_type} prediction: {e}")
            return None
    
    def predict_for_dataframe(self, df):
        """
        Predict cho toàn bộ DataFrame:
        - signal = 1: sử dụng long model
        - signal = 0: sử dụng short model
        Kết quả lưu vào 1 cột predicted_target duy nhất
        """
        if 'long' not in self.models or 'short' not in self.models:
            print("❌ Both long and short models must be loaded")
            return df
        
        try:
            # Chuẩn bị features
            df_prepared = self.prepare_features(df)
            
            # Thêm cột predicted_target
            df_prepared['predicted_target'] = np.nan
            
            # Lấy thông tin từ cả 2 models
            long_info = self.model_infos['long']
            short_info = self.model_infos['short']
            
            long_sequence_len = long_info['sequence_len']
            short_sequence_len = short_info['sequence_len']
            long_features = long_info['features']
            short_features = short_info['features']
            
            print(f"📋 Long model - sequence length: {long_sequence_len}, features: {long_features}")
            print(f"📋 Short model - sequence length: {short_sequence_len}, features: {short_features}")
            
            long_predictions = 0
            short_predictions = 0
            
            # Tính max sequence length để đảm bảo có đủ data
            max_sequence_len = max(long_sequence_len, short_sequence_len)
            
            # Duyệt qua tất cả các điểm
            for i in range(max_sequence_len, len(df_prepared)):
                signal_value = df_prepared['signal'].iloc[i]
                
                if signal_value == 1:
                    # Sử dụng long model
                    if i >= long_sequence_len:
                        seq = df_prepared[long_features].iloc[i-long_sequence_len:i].values
                        predicted_gain = self.predict_max_gain_pct_single(seq, long_features, 'long')
                        
                        if predicted_gain is not None:
                            df_prepared.loc[df_prepared.index[i], 'predicted_target'] = predicted_gain
                            long_predictions += 1
                
                elif signal_value == 0:
                    # Sử dụng short model
                    if i >= short_sequence_len:
                        seq = df_prepared[short_features].iloc[i-short_sequence_len:i].values
                        predicted_gain = self.predict_max_gain_pct_single(seq, short_features, 'short')
                        
                        if predicted_gain is not None:
                            df_prepared.loc[df_prepared.index[i], 'predicted_target'] = predicted_gain
                            short_predictions += 1
            
            total_predictions = long_predictions + short_predictions
            print(f"🎯 Made {long_predictions} predictions with long model (signal=1)")
            print(f"🎯 Made {short_predictions} predictions with short model (signal=0)")
            print(f"🎯 Total predictions made: {total_predictions}")
            
            return df_prepared
            
        except Exception as e:
            print(f"❌ Error in AI prediction process: {e}")
            return df
    
    def get_model_info(self, model_type=None):
        """Lấy thông tin về model(s)"""
        if model_type is None:
            return self.model_infos
        else:
            return self.model_infos.get(model_type, None)
    
    def get_available_models(self):
        """Lấy danh sách các model đã load thành công"""
        return list(self.models.keys())

class RealtimeCryptoPatternAnalyzer:
    def __init__(self):
        self.df = None
        
        # Đỉnh/đáy order=1 (realtime)
        self.realtime_highs = None
        self.realtime_lows = None
        
        # Đỉnh/đáy order=5 (đáng tin cậy)
        self.reliable_highs = None
        self.reliable_lows = None
        
        # AI Model Predictor
        self.ai_predictor = AIModelPredictor()
    
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
                
                # Thêm predicted_target nếu symbol là BTC/USDT và có dữ liệu prediction
                if symbol == 'BTC/USDT' and 'predicted_target' in self.df.columns:
                    if pd.notna(row['predicted_target']):
                        record['predicted_target'] = float(row['predicted_target'])
                    else:
                        record['predicted_target'] = None  # Lưu None thay vì bỏ qua
                
                data_to_save.append(record)
            
            # Lưu vào MongoDB
            if data_to_save:
                # Xóa dữ liệu cũ của symbol này
                delete_result = collection.delete_many({'symbol': symbol})
                print(f"🗑️ Deleted {delete_result.deleted_count} old records for {symbol}")

                # Chèn dữ liệu mới
                result = collection.insert_many(data_to_save)
                
                # Thống kê cho BTC
                if symbol == 'BTC/USDT':
                    predicted_count = sum(1 for record in data_to_save if record.get('predicted_target') is not None)
                    print(f"📊 BTC: Inserted {len(result.inserted_ids)} records, {predicted_count} with AI predictions")
                else:
                    print(f"📊 {symbol}: Inserted {len(result.inserted_ids)} records")
                
                return len(result.inserted_ids)
            else:
                return 0
                
        except Exception as e:
            print(f"❌ Error saving {symbol} to MongoDB: {e}")
            raise Exception(f"Error saving to MongoDB: {e}")
    
    def run_analysis(self, symbol='BTC/USDT', timeframe='1h', count=1000, order_realtime=1, order_reliable=20, threshold=0.01, lookback_window=10):
        """Chạy phân tích và lưu vào MongoDB"""

        print(f"🚀 Starting analysis for {symbol}...")
        
        # Bước 1: Crawl dữ liệu
        self.fetch_data(symbol, timeframe, count)
        print(f"📊 Fetched {len(self.df)} records for {symbol}")

        # Bước 2: Tìm đỉnh và đáy
        peaks_troughs = self.find_peaks_and_troughs(order_realtime=order_realtime, order_reliable=order_reliable)
        print(f"🔍 Found peaks/troughs: Realtime highs={len(peaks_troughs['realtime_highs'])}, Realtime lows={len(peaks_troughs['realtime_lows'])}")

        # Bước 3: Nhận diện các mô hình
        dt_signals = self.detect_realtime_double_top(threshold=threshold, lookback_window=lookback_window)
        db_signals = self.detect_realtime_double_bottom(threshold=threshold, lookback_window=lookback_window)
        hs_signals = self.detect_realtime_head_and_shoulders(threshold=threshold, lookback_window=lookback_window)
        ihs_signals = self.detect_realtime_inverted_head_and_shoulders(threshold=threshold, lookback_window=lookback_window)

        # Bước 4: Kết hợp tín hiệu
        self.combine_signals(dt_signals, db_signals, hs_signals, ihs_signals)
        
        # Đếm số tín hiệu
        buy_signals = (self.df['signal'] == 1).sum()
        sell_signals = (self.df['signal'] == 0).sum()
        print(f"📈 Chart patterns: {buy_signals} buy signals, {sell_signals} sell signals")

        # Bước 5: Chạy AI prediction CHỈ cho BTC (model được train riêng cho BTC)
        if symbol == 'BTC/USDT':
            print("🤖 Running AI predictions for BTC...")
            self.df = self.ai_predictor.predict_for_dataframe(self.df)
            
            # Kiểm tra kết quả AI prediction
            if 'predicted_target' in self.df.columns:
                pred_count = self.df['predicted_target'].notna().sum()
                print(f"🎯 AI Predictions: {pred_count} predictions made")
                
                # In một vài sample predictions để debug
                sample_predictions = self.df[self.df['predicted_target'].notna()][['Date', 'Close', 'signal', 'predicted_target']].tail(5)
                if len(sample_predictions) > 0:
                    print("📋 Sample predictions:")
                    for _, row in sample_predictions.iterrows():
                        print(f"   {row['Date']}: Close={row['Close']:.2f}, Signal={row['signal']}, Predicted={row['predicted_target']:.4f}")
            else:
                print("⚠️ No predicted_target column found after AI prediction")
        else:
            print(f"⏭️ Skipping AI predictions for {symbol} (model only works for BTC)")

        # Bước 6: Lưu vào MongoDB
        saved_count = self.save_to_mongodb(symbol, timeframe)
        
        return saved_count


# ===============================
# FASTAPI ENDPOINTS
# ===============================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Crypto Chart Pattern Analyzer API with AI Predictions",
        "version": "2.0.0",
        "status": "running",
        "features": ["Chart Pattern Detection", "AI Price Prediction for BTC"]
    }

@app.get("/ping")
async def ping():
    """
    Endpoint để cập nhật dữ liệu mới nhất vào MongoDB
    Với AI predictions cho BTC
    """
    try:
        results = {}
        
        # Danh sách symbols để analyze
        symbols = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'SOL/USDT']
        
        for symbol in symbols:
            print(f"\n{'='*50}")
            print(f"Processing {symbol}")
            print(f"{'='*50}")
            
            # Tạo analyzer mới cho mỗi symbol
            analyzer = RealtimeCryptoPatternAnalyzer()
            
            # Chạy phân tích
            analyzer.run_analysis(symbol='BTC/USDT', timeframe='1h', order_realtime=1, order_reliable=20, threshold=0.01, lookback_window=10)
            analyzer.run_analysis(symbol='ETH/USDT', timeframe='1h', order_realtime=1, order_reliable=20, threshold=0.02, lookback_window=10)
            analyzer.run_analysis(symbol='SOL/USDT', timeframe='1h', order_realtime=1, order_reliable=20, threshold=0.02, lookback_window=10)
            analyzer.run_analysis(symbol='XRP/USDT', timeframe='1h', order_realtime=1, order_reliable=20, threshold=0.015, lookback_window=10)
            
        return JSONResponse(
            status_code=200,
            content={
                "message": "DONE",
            }
        )
            
    except Exception as e:
        print(f"❌ Error in ping endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "message": "ERROR",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test MongoDB connection
        client.admin.command('ping')
        
        # Test AI model status
        temp_analyzer = RealtimeCryptoPatternAnalyzer()
        model_status = temp_analyzer.ai_predictor.model is not None
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "mongodb": "connected",
                "ai_model": "loaded" if model_status else "not_loaded",
                "timestamp": datetime.now().isoformat()
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


def start_api_server(host="0.0.0.0", port=8000):
    """Khởi động FastAPI server"""
    print(f"🚀 Starting FastAPI server with AI Integration on {host}:{port}")
    print(f"📖 API Documentation: http://{host}:{port}/docs")
    print(f"🔍 Endpoints:")
    print(f"   - GET /ping - Update latest data to MongoDB (with AI predictions for BTC)")
    print(f"   - GET /health - Health check")
    print(f"🤖 AI Model: Enabled for BTC/USDT predictions")
    print("-" * 50)
    
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_api_server()