# Binary Grey Wolf Optimizer (BGWO) - Feature Selection

Dự án triển khai thuật toán **Binary Grey Wolf Optimizer (BGWO)** cho bài toán **Lựa chọn Đặc trưng (Feature Selection)** trong Machine Learning.

## 📋 Mô tả

Thuật toán BGWO là phiên bản nhị phân của Grey Wolf Optimizer (GWO), được sử dụng để tối ưu hóa việc lựa chọn các đặc trưng quan trọng từ tập dữ liệu. Mục tiêu là tìm ra tập con đặc trưng tối ưu giúp:
- Giảm số lượng đặc trưng (giảm chi phí tính toán)
- Duy trì hoặc cải thiện độ chính xác của mô hình phân loại

## 🎯 Tính năng

- Triển khai thuật toán BGWO với hàm chuyển đổi Sigmoid
- Tích hợp với K-Nearest Neighbors (KNN) classifier
- So sánh hiệu năng giữa mô hình gốc và mô hình tối ưu
- Vẽ biểu đồ hội tụ để theo dõi quá trình tối ưu hóa

## 📦 Yêu cầu hệ thống

- Python 3.7 trở lên
- pip (Python package manager)

## 🚀 Hướng dẫn cài đặt

### Bước 1: Clone repository

```bash
git clone <repository-url>
cd Grey-Wolf-Optimizer---GWO
```

### Bước 2: Tạo môi trường ảo (Virtual Environment)

**Trên Windows:**
```bash
python -m venv venv
```

**Trên Linux/Mac:**
```bash
python3 -m venv venv
```

### Bước 3: Kích hoạt môi trường ảo

**Trên Windows (PowerShell):**
```bash
venv\Scripts\activate
```

**Trên Windows (Command Prompt):**
```bash
venv\Scripts\activate.bat
```

**Trên Linux/Mac:**
```bash
source venv/bin/activate
```

### Bước 4: Cài đặt các thư viện cần thiết

```bash
pip install -r requirements.txt
```

## ▶️ Hướng dẫn chạy chương trình

### 1. Đảm bảo môi trường ảo đã được kích hoạt

**Trên Windows:**
```bash
venv\Scripts\activate
```

**Trên Linux/Mac:**
```bash
source venv/bin/activate
```

### 2. Chạy chương trình

```bash
python main.py
```

## 📊 Kết quả mong đợi

Chương trình sẽ:

1. **Tạo dữ liệu giả lập**: 300 mẫu với 50 đặc trưng (10 đặc trưng có ích, 5 đặc trưng dư thừa)
2. **Chia dữ liệu**: 70% train, 30% test
3. **Chạy thuật toán BGWO**: 
   - 20 con sói (wolves)
   - 30 vòng lặp (iterations)
4. **Hiển thị kết quả**:
   - Số lượng đặc trưng được chọn
   - Danh sách chỉ số đặc trưng được chọn
   - Best Fitness Score
   - So sánh độ chính xác giữa mô hình gốc và mô hình tối ưu
5. **Hiển thị biểu đồ hội tụ**: Biểu đồ theo dõi quá trình tối ưu hóa qua các vòng lặp

## 🔧 Cấu hình

Bạn có thể thay đổi các tham số trong hàm `main()`:

```python
# Số lượng sói và vòng lặp
bgwo = BinaryGWO(num_wolves=20, max_iter=30, data=(X_train, X_test, y_train, y_test))

# Thay đổi kích thước dữ liệu
X, y = make_classification(n_samples=300, n_features=50, n_informative=10, 
                           n_redundant=5, n_classes=2, random_state=42)
```

## 📚 Giải thích thuật toán

### Binary Grey Wolf Optimizer (BGWO)

BGWO mô phỏng hành vi săn mồi của đàn sói xám, với 3 con sói lãnh đạo:
- **Alpha (α)**: Con sói lãnh đạo tốt nhất
- **Beta (β)**: Con sói lãnh đạo thứ hai
- **Delta (δ)**: Con sói lãnh đạo thứ ba

### Quy trình:

1. **Khởi tạo**: Tạo quần thể ngẫu nhiên các vị trí nhị phân (0 hoặc 1)
2. **Đánh giá**: Tính fitness cho mỗi con sói
3. **Cập nhật lãnh đạo**: Xác định Alpha, Beta, Delta dựa trên fitness
4. **Cập nhật vị trí**: 
   - Tính toán vị trí mới dựa trên Alpha, Beta, Delta
   - Chuyển đổi vị trí liên tục sang nhị phân bằng hàm Sigmoid
5. **Lặp lại** cho đến khi đạt số vòng lặp tối đa

### Hàm Fitness

```
Fitness = α × Error_Rate + (1-α) × Feature_Ratio
```

- `Error_Rate`: Tỷ lệ lỗi phân loại (1 - accuracy)
- `Feature_Ratio`: Tỷ lệ đặc trưng được chọn
- `α`: Trọng số (mặc định 0.99)

## 📁 Cấu trúc dự án

```
Grey-Wolf-Optimizer---GWO/
│
├── main.py              # File chính chứa code thuật toán BGWO
├── requirements.txt     # Danh sách các thư viện cần thiết
├── README.md           # File hướng dẫn này
└── venv/               # Thư mục môi trường ảo (không cần commit)
```

## 📦 Các thư viện sử dụng

- **numpy**: Tính toán số học và đại số tuyến tính
- **pandas**: Xử lý và phân tích dữ liệu
- **scikit-learn**: Machine learning (KNN, train/test split, metrics)
- **matplotlib**: Vẽ biểu đồ và trực quan hóa dữ liệu

## 👥 Tác giả

Nhóm 37 - Dựa trên kiến trúc GWO chuẩn

## 📄 License

Dự án này được phát hành dưới giấy phép tự do sử dụng cho mục đích học tập và nghiên cứu.

## 🤝 Đóng góp

Mọi đóng góp đều được chào đón! Hãy tạo một issue hoặc pull request nếu bạn muốn cải thiện dự án.

## 📝 Lưu ý

- Đảm bảo kích hoạt môi trường ảo trước khi chạy chương trình
- Nếu gặp lỗi về thư viện, hãy chạy lại `pip install -r requirements.txt`
- Kết quả có thể khác nhau mỗi lần chạy do tính ngẫu nhiên của thuật toán
