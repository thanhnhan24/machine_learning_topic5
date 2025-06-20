import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import Counter
import joblib
DATA_DIR = r"C:\Users\Duc An Ho\Desktop\Robot-theta\train"
IMAGE_SIZE = (200, 200)  # Resize để đồng đều
SIFT = cv2.SIFT_create()
N_VISUAL_WORDS = 50  # Số từ vựng hình ảnh (Bag of Words)
K = 3  # KMeans cho phân cụm ảnh

# === Bước 1: Trích đặc trưng SIFT từ tất cả ảnh ===
desc_list = []
desc_image_map = []
file_paths = []
true_labels = []

for folder in sorted(os.listdir(DATA_DIR)):
    folder_path = os.path.join(DATA_DIR, folder)
    if not os.path.isdir(folder_path): continue

    for fname in sorted(os.listdir(folder_path)):
        if not fname.endswith(".jpg"): continue
        path = os.path.join(folder_path, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, IMAGE_SIZE)
        keypoints, descriptors = SIFT.detectAndCompute(img, None)

        if descriptors is not None:
            desc_list.extend(descriptors)
            desc_image_map.append(descriptors)
            file_paths.append(path)
            true_labels.append(folder)

# === Bước 2: KMeans lần 1 – tạo từ vựng hình ảnh ===
desc_stack = np.vstack(desc_list)
bow_kmeans = KMeans(n_clusters=N_VISUAL_WORDS, random_state=42)
bow_kmeans.fit(desc_stack)

# === Bước 3: Histogram BoVW cho mỗi ảnh ===
histograms = []

for descriptors in desc_image_map:
    words = bow_kmeans.predict(descriptors)
    hist = np.zeros(N_VISUAL_WORDS)
    for w in words:
        hist[w] += 1
    histograms.append(hist)

histograms = np.array(histograms)

# === Bước 4: PCA để giảm chiều và nhiễu ===
pca = PCA(n_components=30)
reduced_features = pca.fit_transform(histograms)

# === Bước 5: KMeans lần 2 – gom cụm ảnh ===
kmeans = KMeans(n_clusters=K, random_state=42)
clusters = kmeans.fit_predict(reduced_features)
joblib.dump(kmeans, 'kmeans.pkl')
joblib.dump(pca, 'pca.pkl')
joblib.dump(bow_kmeans, 'bow_kmeans.pkl')
# === Bước 6: Hiển thị kết quả ===
import pandas as pd
df = pd.DataFrame({
    'file_path': file_paths,
    'true_label': true_labels,
    'cluster': clusters
})

summary = df.groupby(['true_label', 'cluster']).size().unstack(fill_value=0)
print(summary)
