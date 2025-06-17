import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

# ==================== CONFIGURATION ====================
# Chọn strategy để test: 'long' hoặc 'short'
STRATEGY = 'long'  # Thay đổi thành 'short' để test model short

print(f"🎯 Testing {STRATEGY.upper()} strategy model")

# 1. Load model và scalers đã train
print(f"📥 Loading trained {STRATEGY} model...")

try:
    model = tf.keras.models.load_model(f'models/best_{STRATEGY}_model.h5')
    
    with open(f'models/{STRATEGY}_scaler_X.pkl', 'rb') as f:
        scaler_X = pickle.load(f)
    
    with open(f'models/{STRATEGY}_scaler_y.pkl', 'rb') as f:
        scaler_y = pickle.load(f)
    
    with open(f'models/{STRATEGY}_model_info.pkl', 'rb') as f:
        model_info = pickle.load(f)
    
    print(f"✅ {STRATEGY.upper()} model loaded successfully!")
    print(f"📊 Model info: R² = {model_info['r2_score']:.4f}, RMSE = {model_info['rmse']:.6f}")
    
except FileNotFoundError as e:
    print(f"❌ Error: {STRATEGY} model files not found!")
    print(f"   Make sure you have trained the {STRATEGY} model first")
    print(f"   Missing file: {e.filename}")
    exit()

# 2. Đọc và chuẩn bị dữ liệu
print("\n📊 Loading and preparing data...")
df = pd.read_csv("data.csv")

# Kiểm tra xem có cột 'win' không
if 'win' not in df.columns:
    print("❌ Error: 'win' column not found in data.csv!")
    print("Available columns:", df.columns.tolist())
    exit()

# Tính các chỉ báo (giống như trong training)
df['ma7'] = df['Close'].rolling(window=7).mean()
df['price_vs_ma7'] = (df['Close'] - df['ma7']) / df['ma7']
df['close/open'] = df['Close'] / df['Open'] - 1

# Loại bỏ NaN
df = df.dropna(subset=['price_vs_ma7', 'close/open'])

# 3. Lấy thông tin từ model
sequence_len = model_info['sequence_len']
features = model_info['features']

print(f"📋 Strategy: {STRATEGY}")
print(f"📋 Using sequence length: {sequence_len}")
print(f"📋 Features: {features}")

# Thống kê signals theo strategy
if STRATEGY == 'long':
    target_signals = df[df['signal'] == 1]
    signal_count = (df['signal'] == 1).sum()
    signal_type = "Long signals (signal=1)"
else:  # short
    target_signals = df[df['signal'] == 0]
    signal_count = (df['signal'] == 0).sum()
    signal_type = "Short signals (signal=0)"

print(f"📋 {signal_type}: {signal_count}")
print(f"📋 Using 'win' column for precision calculation")

# 4. Tạo cột kết quả
df['predicted_target'] = np.nan
df['has_prediction'] = 0

# 5. Function để predict
def predict_target_single(model, scaler_X, scaler_y, sequence, features):
    """
    Dự đoán target cho một sequence
    """
    # Chuẩn hóa sequence
    seq_scaled = scaler_X.transform(sequence.reshape(-1, len(features)))
    seq_scaled = seq_scaled.reshape(1, len(sequence), len(features))
    
    # Predict scaled value
    pred_scaled = model.predict(seq_scaled, verbose=0)[0][0]
    
    # Convert back to original scale
    pred_original = scaler_y.inverse_transform([[pred_scaled]])[0][0]
    
    return pred_original

# 6. Duyệt qua các signals để predict
print(f"\n🔍 Predicting targets for {STRATEGY} signals...")
predictions_made = 0

for i in range(sequence_len, len(df)):
    should_predict = False
    
    if STRATEGY == 'long' and df['signal'].iloc[i] == 1:
        should_predict = True
    elif STRATEGY == 'short' and df['signal'].iloc[i] == 0:
        should_predict = True
    
    if should_predict:
        # Tạo sequence từ n phiên trước (không bao gồm phiên hiện tại)
        seq = df[features].iloc[i-sequence_len:i].values
        
        try:
            # Predict target
            predicted_target = predict_target_single(
                model, scaler_X, scaler_y, seq, features
            )
            
            # Lưu kết quả
            df.loc[df.index[i], 'predicted_target'] = predicted_target
            df.loc[df.index[i], 'has_prediction'] = 1
            predictions_made += 1
            
        except Exception as e:
            print(f"⚠️ Error predicting at index {i}: {e}")

print(f"📊 Made {predictions_made} predictions for {STRATEGY} strategy")

# 7. Phân tích kết quả
if predictions_made > 0:
    # Lấy dữ liệu có predictions
    predicted_data = df[df['has_prediction'] == 1].copy()
    
    print(f"\n📊 {STRATEGY.upper()} PREDICTION ANALYSIS:")
    print(f"   Total predictions: {len(predicted_data)}")
    print(f"   Mean predicted_target: {predicted_data['predicted_target'].mean():.6f}")
    print(f"   Std predicted_target: {predicted_data['predicted_target'].std():.6f}")
    print(f"   Min predicted_target: {predicted_data['predicted_target'].min():.6f}")
    print(f"   Max predicted_target: {predicted_data['predicted_target'].max():.6f}")
    
    # Thống kê win
    total_wins = predicted_data['win'].sum()
    overall_win_rate = total_wins / len(predicted_data)
    print(f"   Overall win rate: {overall_win_rate:.4f} ({total_wins}/{len(predicted_data)})")
    
    # 8. Precision analysis theo ngưỡng predicted_target
    print(f"\n📊 Precision analysis for {STRATEGY.upper()} strategy:")
    print("Precision = win=1 count / total count when predicted_target > threshold")
    
    # Tạo các threshold từ min đến max với step 0.1
    min_pred = predicted_data['predicted_target'].min()
    max_pred = predicted_data['predicted_target'].max()
    
    # Tạo thresholds
    thresholds = []
    thresholds.append(min_pred - 0.1)  # Threshold thấp hơn min để capture tất cả
    
    # Thêm các threshold từ 0.1 đến 2.0 với step 0.1
    for t in np.arange(0.1, 2.01, 0.1):
        if min_pred <= t <= max_pred + 0.1:
            thresholds.append(t)
    
    # Thêm một số threshold âm nếu có prediction âm
    if min_pred < 0:
        for t in np.arange(-2.0, 0, 0.2):
            if min_pred <= t < 0:
                thresholds.append(t)
    
    thresholds = sorted(set(thresholds))
    
    stats = []
    
    for threshold in thresholds:
        filtered = predicted_data[predicted_data['predicted_target'] > threshold]
        total_count = len(filtered)
        
        if total_count > 0:
            win_count = filtered['win'].sum()
            precision = win_count / total_count
        else:
            win_count = 0
            precision = np.nan
        
        stats.append({
            'Threshold': round(threshold, 2),
            'Total_Count': total_count,
            'Win_Count': int(win_count),
            'Precision': round(precision, 4) if not np.isnan(precision) else None
        })
    
    # Convert to DataFrame và hiển thị
    precision_df = pd.DataFrame(stats)
    print(precision_df.to_string(index=False))
    
    # Lưu precision analysis
    precision_file = f"precision_by_threshold_{STRATEGY}.csv"
    precision_df.to_csv(precision_file, index=False)
    print(f"\n💾 Precision analysis saved to: {precision_file}")
    
    # 9. Top predictions analysis (với win labels)
    print(f"\n🔝 Top 10 {STRATEGY.upper()} predictions:")
    top_predictions = predicted_data.nlargest(10, 'predicted_target')[
        ['predicted_target', 'win', 'signal']
    ]
    for i, (idx, row) in enumerate(top_predictions.iterrows()):
        win_status = "✅ WIN" if row['win'] == 1 else "❌ LOSS"
        print(f"   {i+1}. Predicted: {row['predicted_target']:.6f}, {win_status}")
    
    print(f"\n🔻 Bottom 10 {STRATEGY.upper()} predictions:")
    bottom_predictions = predicted_data.nsmallest(10, 'predicted_target')[
        ['predicted_target', 'win', 'signal']
    ]
    for i, (idx, row) in enumerate(bottom_predictions.iterrows()):
        win_status = "✅ WIN" if row['win'] == 1 else "❌ LOSS"
        print(f"   {i+1}. Predicted: {row['predicted_target']:.6f}, {win_status}")
    
    # 10. Best threshold analysis
    best_precision_row = precision_df[precision_df['Precision'].notna()].loc[
        precision_df['Precision'].idxmax()
    ]
    print(f"\n🎯 BEST THRESHOLD:")
    print(f"   Threshold: {best_precision_row['Threshold']}")
    print(f"   Precision: {best_precision_row['Precision']}")
    print(f"   Sample size: {best_precision_row['Total_Count']}")
    print(f"   Win count: {best_precision_row['Win_Count']}")

# Bỏ cột has_prediction
df.drop(columns=['has_prediction'], inplace=True)

# 11. Lưu kết quả
output_file = f"data_with_{STRATEGY}_predictions.csv"
df.to_csv(output_file, index=False)
print(f"\n💾 Results saved to: {output_file}")

# 12. Export function để sử dụng sau này
def predict_new_signals(df_new, strategy='long'):
    """
    Function để predict target cho DataFrame mới
    
    Args:
        df_new: DataFrame chứa dữ liệu mới với signals
        strategy: 'long' hoặc 'short'
    
    Returns:
        DataFrame với predictions
    """
    print(f"🔄 Predicting for {strategy} strategy...")
    
    # Load model và scalers
    model = tf.keras.models.load_model(f'models/best_{strategy}_model.h5')
    
    with open(f'models/{strategy}_scaler_X.pkl', 'rb') as f:
        scaler_X = pickle.load(f)
    
    with open(f'models/{strategy}_scaler_y.pkl', 'rb') as f:
        scaler_y = pickle.load(f)
    
    with open(f'models/{strategy}_model_info.pkl', 'rb') as f:
        model_info = pickle.load(f)
    
    # Prepare data
    df_copy = df_new.copy()
    df_copy['predicted_target'] = np.nan
    df_copy['strategy_used'] = strategy
    
    sequence_len = model_info['sequence_len']
    features = model_info['features']
    
    # Predict for each relevant signal
    predictions_made = 0
    for i in range(sequence_len, len(df_copy)):
        should_predict = False
        
        if strategy == 'long' and df_copy['signal'].iloc[i] == 1:
            should_predict = True
        elif strategy == 'short' and df_copy['signal'].iloc[i] == 0:
            should_predict = True
        
        if should_predict:
            seq = df_copy[features].iloc[i-sequence_len:i].values
            pred = predict_target_single(model, scaler_X, scaler_y, seq, features)
            df_copy.loc[df_copy.index[i], 'predicted_target'] = pred
            predictions_made += 1
    
    print(f"✅ Made {predictions_made} predictions for {strategy} strategy")
    return df_copy

print(f"\n🎯 Use predict_new_signals(df, strategy='{STRATEGY}') for future predictions")
print(f"📈 {STRATEGY.upper()} model ready for production use!")

# 13. Hướng dẫn sử dụng
print(f"\n" + "="*60)
print("📋 USAGE INSTRUCTIONS")
print("="*60)
print("1. Để test LONG model:")
print("   - Đặt STRATEGY = 'long'")
print("   - Sẽ test trên signal=1")
print()
print("2. Để test SHORT model:")
print("   - Đặt STRATEGY = 'short'")
print("   - Sẽ test trên signal=0")
print()
print("3. Precision calculation:")
print("   - Precision = (số lượng win=1) / (tổng số mẫu) khi predicted_target > threshold")
print("   - Sử dụng cột 'win' có sẵn trong data.csv")
print()
print("4. Files output:")
print(f"   - Predictions: data_with_{STRATEGY}_predictions.csv")
print(f"   - Precision: precision_by_threshold_{STRATEGY}.csv")
print()
print("5. Production function:")
print("   - predict_new_signals(df, strategy='long') cho long trades")
print("   - predict_new_signals(df, strategy='short') cho short trades")