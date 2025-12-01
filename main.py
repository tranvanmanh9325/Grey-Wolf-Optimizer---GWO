# =============================================================================
# DEMO THUẬT TOÁN BGWO - LỰA CHỌN ĐẶC TRƯNG (FEATURE SELECTION)
# Tác giả: Nhóm 37 (Dựa trên kiến trúc GWO chuẩn)
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. CÁC HÀM HỖ TRỢ (UTILITY FUNCTIONS)
# -----------------------------------------------------------------------------

def sigmoid_transfer(x):
    """
    Hàm chuyển đổi Sigmoid (S-shaped Transfer Function).
    Chuyển đổi giá trị liên tục sang xác suất .
    Công thức: T(x) = 1 / (1 + exp(-10 * (x - 0.5)))
    """
    # Hệ số -10 tạo độ dốc lớn, giúp quyết định dứt khoát.
    # Dịch chuyển -0.5 để căn giữa xác suất tại ngưỡng 0.5.
    return 1 / (1 + np.exp(-10 * (x - 0.5)))

def fitness_function(position, X_train, X_test, y_train, y_test, alpha=0.99):
    """
    Hàm mục tiêu đánh giá chất lượng của tập đặc trưng đã chọn.
    Mục tiêu: Tối thiểu hóa lỗi và số lượng đặc trưng.
    """
    # Chuyển véc-tơ nhị phân thành danh sách chỉ số cột
    cols = [i for i, val in enumerate(position) if val == 1]
    
    # Xử lý trường hợp không chọn đặc trưng nào (tránh lỗi crash)
    if len(cols) == 0:
        return float('inf')
    
    # Lọc dữ liệu theo các đặc trưng đã chọn
    train_x = X_train[:, cols]
    test_x = X_test[:, cols]
    
    # Khởi tạo và huấn luyện bộ phân loại KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train_x, y_train)
    
    # Dự đoán và tính độ chính xác
    pred_y = knn.predict(test_x)
    accuracy = accuracy_score(y_test, pred_y)
    error_rate = 1 - accuracy
    
    # Tính tỷ lệ đặc trưng
    feature_ratio = len(cols) / X_train.shape[1]
    
    # Tính Fitness tổng hợp
    # alpha: trọng số cho error rate (thường là 0.99)
    # (1-alpha): trọng số cho số lượng đặc trưng (0.01)
    fitness = alpha * error_rate + (1 - alpha) * feature_ratio
    
    return fitness

# -----------------------------------------------------------------------------
# 2. LỚP THUẬT TOÁN BGWO (CORE ALGORITHM)
# -----------------------------------------------------------------------------

class BinaryGWO:
    def __init__(self, num_wolves, max_iter, data):
        self.num_wolves = num_wolves
        self.max_iter = max_iter
        self.X_train, self.X_test, self.y_train, self.y_test = data
        self.dim = self.X_train.shape[1]
        
        # Khởi tạo quần thể ngẫu nhiên (0 hoặc 1)
        self.wolves = np.random.randint(0, 2, (self.num_wolves, self.dim))
        
        # Khởi tạo Alpha, Beta, Delta
        self.alpha_pos = np.zeros(self.dim)
        self.alpha_score = float('inf')
        
        self.beta_pos = np.zeros(self.dim)
        self.beta_score = float('inf')
        
        self.delta_pos = np.zeros(self.dim)
        self.delta_score = float('inf')
        
        # Lưu lịch sử hội tụ để vẽ biểu đồ
        self.convergence_curve = []

    def optimize(self):
        print(f"[*] Bắt đầu tối ưu BGWO: {self.num_wolves} sói, {self.max_iter} vòng lặp, {self.dim} chiều.")
        
        for t in range(self.max_iter):
            # --- BƯỚC 1: Đánh giá Fitness và Cập nhật Lãnh đạo ---
            for i in range(self.num_wolves):
                # Tính fitness cho sói thứ i
                fit = fitness_function(self.wolves[i], self.X_train, self.X_test, self.y_train, self.y_test)
                
                # Cập nhật Alpha, Beta, Delta
                if fit < self.alpha_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy()
                    self.beta_score = self.alpha_score
                    self.beta_pos = self.alpha_pos.copy()
                    self.alpha_score = fit
                    self.alpha_pos = self.wolves[i].copy()
                elif fit < self.beta_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy()
                    self.beta_score = fit
                    self.beta_pos = self.wolves[i].copy()
                elif fit < self.delta_score:
                    self.delta_score = fit
                    self.delta_pos = self.wolves[i].copy()
            
            # Lưu lại kết quả tốt nhất của vòng lặp
            self.convergence_curve.append(self.alpha_score)
            print(f"    > Vòng lặp {t+1}/{self.max_iter} | Best Fitness: {self.alpha_score:.5f}")
            
            # --- BƯỚC 2: Cập nhật Vị trí ---
            a = 2 - t * (2 / self.max_iter) # Tham số a giảm tuyến tính từ 2 về 0
            
            for i in range(self.num_wolves):
                for d in range(self.dim):
                    # Tính toán dựa trên Alpha
                    r1, r2 = np.random.random(), np.random.random()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * self.alpha_pos[d] - self.wolves[i, d])
                    X1 = self.alpha_pos[d] - A1 * D_alpha
                    
                    # Tính toán dựa trên Beta
                    r1, r2 = np.random.random(), np.random.random()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * self.beta_pos[d] - self.wolves[i, d])
                    X2 = self.beta_pos[d] - A2 * D_beta
                    
                    # Tính toán dựa trên Delta
                    r1, r2 = np.random.random(), np.random.random()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * self.delta_pos[d] - self.wolves[i, d])
                    X3 = self.delta_pos[d] - A3 * D_delta
                    
                    # Vị trí liên tục trung bình (Continuous Step)
                    X_continuous = (X1 + X2 + X3) / 3
                    
                    # --- BƯỚC QUAN TRỌNG: CHUYỂN ĐỔI SANG NHỊ PHÂN ---
                    # Sử dụng hàm Sigmoid để lấy xác suất
                    prob = sigmoid_transfer(X_continuous)
                    
                    # Stochastic Thresholding (Ngưỡng ngẫu nhiên)
                    if np.random.random() < prob:
                        self.wolves[i, d] = 1
                    else:
                        self.wolves[i, d] = 0
        
        return self.alpha_pos, self.alpha_score, self.convergence_curve

# -----------------------------------------------------------------------------
# 3. CHẠY DEMO (MAIN DRIVER)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # A. Tạo dữ liệu giả lập (Synthetic Dataset)
    # 300 mẫu, 50 đặc trưng (trong đó chỉ có 10 đặc trưng thực sự có ích)
    print("=== TẠO DỮ LIỆU GIẢ LẬP ===")
    X, y = make_classification(n_samples=300, n_features=50, n_informative=10, 
                               n_redundant=5, n_classes=2, random_state=42)
    
    # Chia tập Train/Test (70/30)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"Kích thước tập train: {X_train.shape}, Tập test: {X_test.shape}")
    
    # B. Chạy thuật toán BGWO
    # Cấu hình: 20 con sói, 30 vòng lặp
    bgwo = BinaryGWO(num_wolves=20, max_iter=30, data=(X_train, X_test, y_train, y_test))
    best_pos, best_score, curve = bgwo.optimize()
    
    # C. Phân tích kết quả
    selected_indices = [i for i, val in enumerate(best_pos) if val == 1]
    
    print("\n" + "="*40)
    print("KẾT QUẢ TỐI ƯU HÓA")
    print("="*40)
    print(f"Tổng số đặc trưng ban đầu: {X.shape}")
    print(f"Số lượng đặc trưng được chọn: {len(selected_indices)}")
    print(f"Các đặc trưng được chọn (indices): {selected_indices}")
    print(f"Best Fitness Score: {best_score:.6f}")
    
    # D. So sánh hiệu năng (Baseline vs Optimized)
    # 1. Baseline: Dùng tất cả đặc trưng
    knn_base = KNeighborsClassifier(n_neighbors=5)
    knn_base.fit(X_train, y_train)
    acc_base = accuracy_score(y_test, knn_base.predict(X_test))
    
    # 2. Optimized: Dùng đặc trưng do BGWO chọn
    knn_opt = KNeighborsClassifier(n_neighbors=5)
    knn_opt.fit(X_train[:, selected_indices], y_train)
    acc_opt = accuracy_score(y_test, knn_opt.predict(X_test[:, selected_indices]))
    
    print("-" * 40)
    print(f"Độ chính xác gốc (50 features): {acc_base * 100:.2f}%")
    print(f"Độ chính xác sau BGWO ({len(selected_indices)} features): {acc_opt * 100:.2f}%")
    print("-" * 40)
    
    # E. Vẽ biểu đồ hội tụ
    plt.figure(figsize=(10, 6))
    plt.plot(curve, color='red', marker='o', markevery=5)
    plt.title('Biểu đồ Hội tụ của BGWO cho Lựa chọn Đặc trưng')
    plt.xlabel('Vòng lặp (Iteration)')
    plt.ylabel('Giá trị Fitness (Càng nhỏ càng tốt)')
    plt.grid(True)
    plt.show()
    