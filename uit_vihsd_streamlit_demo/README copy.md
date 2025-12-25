
# Hydro Web Demo

Web app (Flask) để nhập **ngày/giờ cần dự báo** và trả về **mực nước thượng lưu** theo 3 mô hình:
- **XGBoost** (`xgb_model.json`, `features_used*.json`)
- **Decision Tree** (`dtree_model.pkl`, `features_used*.json` — pipeline gồm imputer + tree)
- **CNN+LSTM** (`cnn_lstm_model_seq6_28d.pth`, `seq_scaler_28d_stride1_6ch.pkl`, `y_scaler_28d_stride1_6ch.pkl`)

Dữ liệu inference: `data/data_thuydien_enriched.csv`.

## Cấu trúc
```
hydro_web_demo/
  app.py
  requirements.txt
  models/
    cnn_lstm_model_seq6_28d.pth
    seq_scaler_28d_stride1_6ch.pkl
    y_scaler_28d_stride1_6ch.pkl
    xgb_model.json
    dtree_model.pkl
    features_used.json            (nếu dùng chung cho cả XGB & DT, copy file này 2 lần hoặc sửa app.py)
    features_used_xgb.json        (tùy chọn: nếu tách riêng)
    features_used_dtree.json      (tùy chọn: nếu tách riêng)
  data/
    data_thuydien_enriched.csv
  utils/
    features.py
    models.py
    inference.py
  templates/
    index.html
  static/
    styles.css
```

## Chạy thử
```bash
cd hydro_web_demo
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# Copy các file model & enriched vào thư mục models/ và data/
#   models/xgb_model.json
#   models/dtree_model.pkl
#   models/cnn_lstm_model_seq6_28d.pth
#   models/seq_scaler_28d_stride1_6ch.pkl
#   models/y_scaler_28d_stride1_6ch.pkl
#   models/features_used.json (hoặc features_used_xgb.json & features_used_dtree.json)
#   data/data_thuydien_enriched.csv

python app.py
# Mở http://localhost:8000
```

## Ghi chú
- App **anchor** vào bản ghi gần nhất trước thời điểm yêu cầu, rồi tính đặc trưng **strictly-past 28 ngày** cho XGB/DTree; với CNN+LSTM, tạo **chuỗi (L,6)** theo scaler và lưới đều kết thúc tại anchor.
- Nếu bạn chỉ có **một** `features_used.json`, app sẽ dùng file đó cho **cả 2** mô hình XGB và Decision Tree (xem `app.py`).
- Nếu `torch` không khả dụng, phần CNN+LSTM sẽ báo lỗi — nhưng XGB/DTree vẫn chạy bình thường.

# (1) kiểm tra python
python --version
# nếu chưa cài, cài từ python.org hoặc dùng winget

# (2) vào thư mục dự án
cd C:\path\to\hydro_web_demo

# (3) tạo venv bằng py launcher
py -3.11 -m venv .venv
# hoặc: python -m venv .venv

# (4) nếu cần: cho phép chạy script activation (chỉ 1 lần)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# (5) activate
.\.venv\Scripts\Activate.ps1

# (6) cập nhật pip và cài requirements
python -m pip install --upgrade pip
pip install -r requirements.txt

# (7) copy các file model + data vào folders models/ và data/

# (8) chạy app
python app.py
# mở http://localhost:8000

