# Bitcoin Price Prediction Tool 🚀

Ứng dụng web dự báo giá Bitcoin sử dụng các model Machine Learning đã được train sẵn (ARIMA, SVM Linear, SVM RBF).

## ✨ Tính Năng

- 📊 **Biểu đồ tương tác**: Hiển thị giá lịch sử và dự đoán
- 🤖 **Đa model**: Hỗ trợ ARIMA, SVM Linear, SVM RBF
- 📈 **Metrics hiệu suất**: MAPE, Accuracy, MAE
- 🌐 **API tự động**: Lấy dữ liệu từ CoinGecko
- 📱 **Responsive**: Giao diện thích ứng mọi thiết bị
- 💾 **Export kết quả**: Xuất dữ liệu JSON

## 🏗️ Cấu Trúc Dự Án

```
app/
├── config/
│   └── model_config.json    # Cấu hình model
├── models/                  # Model đã train (cần thêm)
│   ├── arima_model.pkl
│   ├── svm_linear.pkl
│   ├── svm_rbf.pkl
│   └── scaler.pkl
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
├── templates/
│   └── index.html
├── main.py                  # Ứng dụng chính
└── data/                    # Thư mục dữ liệu (tùy chọn)
```

## 🚀 Cài Đặt

### 1. Clone Repository

```bash
git clone <repository-url>
cd bitcoin-prediction-tool
```

### 2. Tạo Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# MacOS/Linux
source venv/bin/activate
```

### 3. Cài Đặt Dependencies

```bash
pip install -r requirements.txt
```

### 4. Chuẩn Bị Models

Đặt các file model đã train vào thư mục `app/models/`:
- `arima_model.pkl`
- `svm_linear.pkl`
- `svm_rbf.pkl`
- `scaler.pkl`

### 5. Chạy Ứng Dụng

```bash
cd app
python main.py
```

Truy cập: `http://localhost:5000`

## 📋 Yêu Cầu Hệ Thống

- Python 3.8+
- Internet connection (để lấy dữ liệu từ API)
- Model files (.pkl)

## 🎯 Cách Sử Dụng

### 1. Chọn Khoảng Thời Gian
- Thiết lập ngày bắt đầu và kết thúc
- Hệ thống sẽ dự báo cho khoảng thời gian này

### 2. Chọn Models
- ✅ ARIMA: Time series forecasting
- ✅ SVM Linear: Support Vector Machine với Linear kernel
- ✅ SVM RBF: Support Vector Machine với RBF kernel

### 3. Lấy Dữ Liệu
- Click "Fetch Data" để lấy dữ liệu mới nhất
- Dữ liệu từ CoinGecko API

### 4. Chạy Dự Báo
- Click "Predict" để chạy các model đã chọn
- Xem kết quả trên biểu đồ và bảng metrics

### 5. Export Kết Quả
- Click "Export Results" để tải file JSON
- Chứa toàn bộ dữ liệu và kết quả dự báo

## 📊 Performance Metrics

### ARIMA
- **MAPE** (Mean Absolute Percentage Error)
  - Excellent: ≤ 10%
  - Good: 10-25%
  - Poor: > 25%

### SVM Models
- **Accuracy**: Độ chính xác dự đoán
  - Excellent: ≥ 80%
  - Good: 60-80%
  - Poor: < 60%
- **MAE** (Mean Absolute Error): Sai số tuyệt đối trung bình

## 🎨 Giao Diện

- **Header**: Logo Bitcoin với animation
- **Left Panel**: Controls và settings
- **Right Panel**: Biểu đồ và metrics
- **Status Bar**: Trạng thái và thời gian cập nhật
- **Loading Overlay**: Hiệu ứng loading

## 🔧 Cấu Hình

Chỉnh sửa `config/model_config.json`:

```json
{
  "models": {
    "arima": {
      "path": "./models/arima_model.pkl",
      "window_size": 4
    },
    "svm": {
      "linear": {
        "path": "./models/svm_linear.pkl",
        "scaler_path": "./models/scaler.pkl"
      }
    }
  },
  "data_source": {
    "api_url": "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart",
    "days": "max",
    "currency": "usd"
  }
}
```

## 🐛 Troubleshooting

### Model Not Found
- Kiểm tra đường dẫn trong `model_config.json`
- Đảm bảo file model tồn tại trong thư mục `models/`

### API Error
- Kiểm tra kết nối internet
- CoinGecko API có thể bị giới hạn rate limit

### Dependencies Error
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

## 🤝 Đóng Góp

1. Fork repository
2. Tạo feature branch: `git checkout -b feature/AmazingFeature`
3. Commit changes: `git commit -m 'Add AmazingFeature'`
4. Push to branch: `git push origin feature/AmazingFeature`
5. Open Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📞 Contact

Your Name - your.email@example.com

Project Link: [https://github.com/yourusername/bitcoin-prediction-tool](https://github.com/yourusername/bitcoin-prediction-tool)

---

Made with ❤️ for Bitcoin prediction 