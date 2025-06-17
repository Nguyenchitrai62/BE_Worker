# 🧠 BE Worker – Xử lý tín hiệu & AI dự đoán (rule-based + LSTM)

Đây là thành phần backend chuyên xử lý tín hiệu giao dịch theo thời gian thực, bao gồm:

- Phân tích mẫu hình giá bằng rule-based (Double Top / Double Bottom…)
- Dự đoán biến động giá bằng mô hình LSTM hồi quy
- Cập nhật kết quả phân tích và dự đoán vào cơ sở dữ liệu MongoDB

Ứng dụng này hỗ trợ các tài sản tiền điện tử hàng đầu: **BTC, ETH, SOL, XRP** và có khả năng mở rộng linh hoạt.

---

## 🚀 Cách hoạt động

- Server API sử dụng `FastAPI`, chạy từ file `app_VIP.py`
- Khi gọi endpoint `GET /ping`, server sẽ:
  1. Lấy dữ liệu OHLCV mới nhất từ sàn Binance thông qua thư viện `ccxt`
  2. Chạy thuật toán rule-based để phát hiện mẫu hình giá (Double Top / Bottom…)
  3. Sử dụng mô hình LSTM hồi quy để xác nhận tín hiệu
  4. Ghi kết quả phân tích và dự đoán vào MongoDB

---

## 🧪 Công nghệ sử dụng

- **FastAPI** – Web framework nhẹ, nhanh
- **Uvicorn** – ASGI server cho FastAPI
- **ccxt** – Lấy dữ liệu giá từ các sàn giao dịch
- **NumPy, Pandas** – Xử lý dữ liệu tài chính
- **SciPy** – Tìm điểm cực trị trong chuỗi giá
- **PyTorch (local)** – Mô hình học sâu (LSTM)
- **pymongo** – Kết nối và ghi dữ liệu vào MongoDB
- **dotenv** – Quản lý cấu hình môi trường

---

## ⚙️ Cài đặt & chạy local

### Yêu cầu:
- Python >= 3.9
- Có tài khoản MongoDB và file `.env` cấu hình đúng

### Cài đặt:

```bash
# 1. Clone dự án
git clone https://github.com/Nguyenchitrai62/BE_Worker
cd BE_Worker

# 2. Cài đặt thư viện
pip install -r requirements.txt

# 3. Chạy server xử lý
uvicorn app_VIP:app --reload --port 8000
```

---

## 🧩 Các thành phần khác của dự án

- 🌐 **Frontend – Giao diện trực quan người dùng**  
  👉 [https://github.com/Nguyenchitrai62/FE_Crypto_Intelligence](https://github.com/Nguyenchitrai62/FE_Crypto_Intelligence)

- 🔧 **BE Web – API & Notification Service**  
  👉 [https://github.com/Nguyenchitrai62/BE_WEB_Crypto](https://github.com/Nguyenchitrai62/BE_WEB_Crypto)
