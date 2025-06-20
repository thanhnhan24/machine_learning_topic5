
# 🤖 Hệ thống điều khiển cánh tay robot phân loại vật thể

Dự án xây dựng hệ thống điều khiển cánh tay robot **xArm Lite6**, sử dụng camera **Intel RealSense D435i** để nhận diện vật thể thông qua mã ArUco, phân loại bằng **BoVW (SIFT + KMeans + PCA)**, và gắp vật đến đúng pallet. Hệ thống có GUI điều khiển, chế độ calibration và chạy dòng lệnh (CLI).

## 🗂️ Cấu trúc dự án

```
.
├── app.py                  # Giao diện điều khiển chính (GUI, PySide2)
├── box_grabber.py          # Chế độ chạy không giao diện, tự động phát hiện và gắp vật
├── kmean_train.py          # Huấn luyện mô hình phân loại vật thể bằng BoVW
├── pallett_calib.py        # Hiệu chỉnh vị trí các pallet (calibration)
├── theta_cal.json          # File cấu hình vị trí và hiệu chỉnh pallet
├── ui_source.py            # Giao diện GUI được chuyển từ Qt Designer
├── *.pkl                   # Các file mô hình đã huấn luyện (BoVW, PCA, KMeans)
└── train/                  # Thư mục chứa dữ liệu huấn luyện
```

## 🚀 Hướng dẫn sử dụng

### 1. Cài đặt thư viện cần thiết

```bash
pip install opencv-python pyrealsense2 PySide2 scikit-learn joblib matplotlib pandas
```

### 2. Huấn luyện mô hình phân loại

```bash
python kmean_train.py
```

- Đặt ảnh vào thư mục `train/`, phân theo các nhãn `id_1`, `id_2`, `id_3`
- Sinh ra các file: `bow_kmeans.pkl`, `pca.pkl`, `kmeans.pkl`

### 3. Giao diện GUI điều khiển chính

```bash
python app.py
```

- Kết nối robot + camera
- Cấu hình pallet
- Load mô hình và bắt đầu phân loại gắp vật thể

### 4. Chạy không giao diện (CLI)

```bash
python box_grabber.py
```

- Nhận diện ArUco, crop box, phân loại, và ra lệnh gắp tự động

### 5. Calibration vị trí pallet

```bash
python pallett_calib.py
```

- Di chuyển robot đến pallet → camera phát hiện box → tự động hiệu chỉnh tọa độ
- Nhấn `y` để lưu vào `theta_cal.json`

## 🧠 Mô hình phân loại vật thể

- **SIFT** để trích đặc trưng ảnh
- **KMeans** tạo từ điển hình ảnh (BoVW)
- **PCA** giảm chiều vector histogram
- **KMeans** phân cụm vật thể thành `id_1`, `id_2`, `id_3`

## 📷 Nhận diện và định vị

- Sử dụng mã ArUco 4x4_50 và 5x5_50
- Tính toán góc và tâm box để robot căn chỉnh và gắp chính xác
- Chờ box ổn định trước khi ra lệnh di chuyển

## ⚙️ Cấu hình hệ thống

- `theta_cal.json`: lưu thông tin tọa độ và hiệu chỉnh từng pallet (x, y, z, yaw)

## 📝 Nhóm thực hiện

- **Nguyễn Thanh Nhân**
- **Nguyễn Vũ Huy Khôi**
- **Hồ Đức An**
- Đề tài: Tự động điều khiển cánh tay robot gắp và phân loại vật thể
- HUTECH 2025 – Môn: Máy học ứng dụng

## 📌 Ghi chú

- Cần đảm bảo camera RealSense và robot xArm Lite6 đã kết nối đúng
- Nếu không có phần cứng có thể thử nghiệm bằng ảnh hoặc video đã ghi
