import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pickle
import os

# ==================== CONFIGURATION ====================
# Chọn strategy: 'long', 'short'
STRATEGY = 'short'  

print(f"🎯 Training strategy: {STRATEGY.upper()}")

# 1. Đọc dữ liệu
df = pd.read_csv("data.csv")
df = df.dropna(subset=['price_vs_ma7', 'close/open', 'max_gain_pct', 'max_loss_pct'])

sequence_len = 15
features = ['price_vs_ma7', 'close/open']
X, y = [], []

# 2. Tạo chuỗi đầu vào theo strategy
print("📊 Creating sequences...")

def create_sequences(df, strategy):
    """
    Tạo sequences theo strategy
    """
    X_data, y_data = [], []
    
    for i in range(sequence_len, len(df)):
        seq = df[features].iloc[i-sequence_len:i].values
        
        if strategy == 'long' and df['signal'].iloc[i] == 1:
            # Long: signal=1, target = max_gain_pct - max_loss_pct
            target = df['max_gain_pct'].iloc[i] - df['max_loss_pct'].iloc[i]
            target = np.clip(target, -3, 3)
            X_data.append(seq)
            y_data.append(target)
            
        elif strategy == 'short' and df['signal'].iloc[i] == 0:
            # Short: signal=0, target = max_loss_pct - max_gain_pct  
            target = df['max_loss_pct'].iloc[i] - df['max_gain_pct'].iloc[i]
            target = np.clip(target, -3, 3)
            X_data.append(seq)
            y_data.append(target)
             
    return np.array(X_data), np.array(y_data)

X, y = create_sequences(df, STRATEGY)

print(f"📊 Total {STRATEGY} samples: {len(X)}")
print(f"📊 Target statistics:")
print(f"   Mean target: {np.mean(y):.4f}")
print(f"   Std target: {np.std(y):.4f}")
print(f"   Min target: {np.min(y):.4f}")
print(f"   Max target: {np.max(y):.4f}")

if len(X) == 0:
    print("❌ No samples found! Check your data and strategy settings.")
    exit()

# 3. Chuẩn hóa features (X)
scaler_X = StandardScaler()
X_reshaped = X.reshape(-1, X.shape[-1])
X_scaled = scaler_X.fit_transform(X_reshaped)
X_scaled = X_scaled.reshape(X.shape)

# Chuẩn hóa target (y) để training ổn định hơn
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

print(f"📊 After scaling:")
print(f"   X mean: {np.mean(X_scaled, axis=(0,1))}")
print(f"   X std: {np.std(X_scaled, axis=(0,1))}")
print(f"   y_scaled mean: {np.mean(y_scaled):.4f}")
print(f"   y_scaled std: {np.std(y_scaled):.4f}")

# 4. Chia train/test theo thời gian
split_index = int(len(X_scaled) * 0.8)
X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
y_train, y_test = y_scaled[:split_index], y_scaled[split_index:]
y_test_original = y[split_index:]

print(f"📊 Train: {len(X_train)} samples")
print(f"📊 Test: {len(X_test)} samples")

# 5. Model LSTM
print(f"\n🏗️ Creating LSTM model for {STRATEGY} strategy...")

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(32, input_shape=(sequence_len, len(features)), 
                        return_sequences=False, dropout=0.2, recurrent_dropout=0.1),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(8, activation='relu'),  
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss='mse',
    metrics=['mae']
)

print(f"\n🏗️ Model Architecture for {STRATEGY}:")
model.summary()

# 6. Callbacks
os.makedirs('models', exist_ok=True)
best_model_path = f'models/best_{STRATEGY}_model.h5'

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    best_model_path,
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=0.00001,
    verbose=1
)

# 7. Training
print(f"\n🚀 Training {STRATEGY} model...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint, reduce_lr],
    verbose=1
)

# 8. Load model tốt nhất
print(f"\n📥 Loading best {STRATEGY} model...")
best_model = tf.keras.models.load_model(best_model_path)

# 9. Đánh giá model
print("\n" + "="*60)
print(f"📊 {STRATEGY.upper()} MODEL EVALUATION")
print("="*60)

y_pred_scaled = best_model.predict(X_test, verbose=0).flatten()
y_pred_original = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

mse = mean_squared_error(y_test_original, y_pred_original)
mae = mean_absolute_error(y_test_original, y_pred_original)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_original, y_pred_original)

print(f"📈 {STRATEGY.upper()} Regression Metrics:")
print(f"   MSE: {mse:.6f}")
print(f"   RMSE: {rmse:.6f}")
print(f"   MAE: {mae:.6f}")
print(f"   R² Score: {r2:.4f}")

print(f"\n📊 Prediction vs Actual:")
print(f"   Actual mean: {np.mean(y_test_original):.6f}")
print(f"   Predicted mean: {np.mean(y_pred_original):.6f}")
print(f"   Actual std: {np.std(y_test_original):.6f}")
print(f"   Predicted std: {np.std(y_pred_original):.6f}")

# 10. Phân tích predictions
residuals = y_test_original - y_pred_original
print(f"\n📊 Prediction Analysis:")
print(f"   Mean residual: {np.mean(residuals):.6f}")
print(f"   Std residual: {np.std(residuals):.6f}")
print(f"   Max error: {np.max(np.abs(residuals)):.6f}")

# Top và bottom predictions
top_indices = np.argsort(y_pred_original)[-5:]
bottom_indices = np.argsort(y_pred_original)[:5]

print(f"\n🔝 Top 5 predicted {STRATEGY} targets:")
for i, idx in enumerate(reversed(top_indices)):
    print(f"   {i+1}. Predicted: {y_pred_original[idx]:.6f}, Actual: {y_test_original[idx]:.6f}")

print(f"\n🔻 Bottom 5 predicted {STRATEGY} targets:")
for i, idx in enumerate(bottom_indices):
    print(f"   {i+1}. Predicted: {y_pred_original[idx]:.6f}, Actual: {y_test_original[idx]:.6f}")

# 11. Save scalers và model info
scaler_X_path = f'models/{STRATEGY}_scaler_X.pkl'
scaler_y_path = f'models/{STRATEGY}_scaler_y.pkl'

with open(scaler_X_path, 'wb') as f:
    pickle.dump(scaler_X, f)

with open(scaler_y_path, 'wb') as f:
    pickle.dump(scaler_y, f)

model_info = {
    'strategy': STRATEGY,
    'sequence_len': sequence_len,
    'features': features,
    'target_formula': {
        'long': 'max_gain_pct - max_loss_pct',
        'short': 'max_loss_pct - max_gain_pct',
        'both': 'long + short combined'
    }[STRATEGY],
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'mse': float(mse),
    'mae': float(mae),
    'rmse': float(rmse),
    'r2_score': float(r2),
    'target_mean': float(np.mean(y)),
    'target_std': float(np.std(y))
}

info_path = f'models/{STRATEGY}_model_info.pkl'
with open(info_path, 'wb') as f:
    pickle.dump(model_info, f)

print(f"\n💾 {STRATEGY.upper()} Files saved:")
print(f"   - {best_model_path}")  
print(f"   - {scaler_X_path}")
print(f"   - {scaler_y_path}")
print(f"   - {info_path}")

# 12. Production function để predict
def predict_target(model, scaler_X, scaler_y, df_new, strategy, sequence_len=sequence_len):
    """
    Dự đoán target cho strategy được chọn
    
    Args:
        model: Trained LSTM model
        scaler_X: Fitted StandardScaler for features
        scaler_y: Fitted StandardScaler for target
        df_new: DataFrame with new data
        strategy: 'long', 'short', hoặc 'both'
        sequence_len: Sequence length used in training
    
    Returns:
        DataFrame with predictions
    """
    results = []
    
    for i in range(sequence_len, len(df_new)):
        should_predict = False
        signal_type = None
        
        if strategy == 'long' and df_new['signal'].iloc[i] == 1:
            should_predict = True
            signal_type = 'long'
        elif strategy == 'short' and df_new['signal'].iloc[i] == 0:
            should_predict = True
            signal_type = 'short'
        elif strategy == 'both':
            if df_new['signal'].iloc[i] == 1:
                should_predict = True
                signal_type = 'long'
            elif df_new['signal'].iloc[i] == 0:
                should_predict = True
                signal_type = 'short'
        
        if should_predict:
            # Create sequence
            seq = df_new[features].iloc[i-sequence_len:i].values
            seq_scaled = scaler_X.transform(seq.reshape(-1, len(features))).reshape(1, sequence_len, len(features))
            
            # Predict scaled value
            pred_scaled = model.predict(seq_scaled, verbose=0)[0][0]
            
            # Convert back to original scale
            pred_original = scaler_y.inverse_transform([[pred_scaled]])[0][0]
            
            results.append({
                'index': i,
                'timestamp': df_new.index[i] if hasattr(df_new.index, 'to_list') else i,
                'signal': df_new['signal'].iloc[i],
                'signal_type': signal_type,
                'predicted_target': pred_original,
                'strategy': strategy
            })
    
    return pd.DataFrame(results)

print(f"\n✅ {STRATEGY.upper()} model ready!")
print(f"🎯 Use predict_target() function for new predictions")
print(f"📈 Model performance: R² = {r2:.4f}, RMSE = {rmse:.6f}")

# 13. Training history summary
print(f"\n📈 Training History Summary:")
print(f"   Final train loss: {history.history['loss'][-1]:.6f}")
print(f"   Final val loss: {history.history['val_loss'][-1]:.6f}")
print(f"   Best val loss: {min(history.history['val_loss']):.6f}")
print(f"   Final train MAE: {history.history['mae'][-1]:.6f}")
print(f"   Final val MAE: {history.history['val_mae'][-1]:.6f}")

# 14. Hướng dẫn sử dụng
print(f"\n" + "="*60)
print("📋 USAGE INSTRUCTIONS")
print("="*60)
print("1. Để train model LONG:")
print("   - Đặt STRATEGY = 'long'")
print("   - Model sẽ dùng signal=1 và target = max_gain_pct - max_loss_pct")
print()
print("2. Để train model SHORT:")
print("   - Đặt STRATEGY = 'short'") 
print("   - Model sẽ dùng signal=0 và target = max_loss_pct - max_gain_pct")
print()
print("3. Để train model BOTH:")
print("   - Đặt STRATEGY = 'both'")
print("   - Model sẽ dùng cả signal=0 và signal=1")
print()
print("4. Files được lưu với tên strategy:")
print(f"   - Model: models/best_{STRATEGY}_model.h5")
print(f"   - Scalers: models/{STRATEGY}_scaler_X.pkl, models/{STRATEGY}_scaler_y.pkl")
print(f"   - Info: models/{STRATEGY}_model_info.pkl")