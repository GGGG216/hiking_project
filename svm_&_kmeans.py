import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.decomposition import PCA  # Add PCA for dimensionality reduction
import seaborn as sns

def load_data(image_folder, image_size=(512,512)):
    images = []
    labels = []
    for file in os.listdir(image_folder):
        if file.endswith('.npy'): 
           
            npy_path = os.path.join(image_folder, file)
            image = np.load(npy_path)


            label_file = file.replace('.npy', '.txt')
            label_path = os.path.join(image_folder, label_file)
            if os.path.exists(label_path):  
                with open(label_path, 'r') as f:
                    label = int(f.read().strip())
                    images.append(image)
                    labels.append(label)
            # Note: This reshape operation seems incorrect and unnecessary here
            # image.reshape(-1, 2)  # This doesn't modify the image in place
    return np.array(images), np.array(labels)


def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


def plot_kmeans_clusters(images, n_clusters=5):
    
    # Assuming images shape is (n_samples, height, width) or (n_samples, height, width, channels)
    n_samples = images.shape[0]
    flattened_images = images.reshape(n_samples, -1)  # Flatten each image into a 1D array

  
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(flattened_images)
    pca = PCA(n_components=2, random_state=42)
    reduced_features = pca.fit_transform(flattened_images)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=clusters, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Cluster')
    plt.title("KMeans Clustering Results (PCA Reduced)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()


def main():

    image_folder = './data/depth_maps/train'
    print("加载数据...")
    images, labels = load_data(image_folder)
    print(f"加载图片数量: {len(images)}, 标签数量: {len(labels)}")
    print("划分数据集...")
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    print("使用KMeans进行预测...")
    kmeans = KMeans(n_clusters=5, init="k-means++", random_state=42)
    # Flatten images for KMeans
    n_samples = X_train.shape[0]
    X_train_flattened = X_train.reshape(n_samples, -1)
    kmeans.fit(X_train_flattened)
    # Flatten test images for prediction
    X_test_flattened = X_test.reshape(X_test.shape[0], -1)
    y_pred_kmeans = kmeans.predict(X_test_flattened)
    cluster_to_label = {}
    for cluster in range(kmeans.n_clusters):
        cluster_indices = np.where(kmeans.labels_ == cluster)[0]
        cluster_labels = y_train[cluster_indices]
        if len(cluster_labels) > 0:
            cluster_to_label[cluster] = np.bincount(cluster_labels).argmax()

    y_pred_kmeans_mapped = np.array([cluster_to_label[cluster] for cluster in y_pred_kmeans])
    print("KMeans分类报告:")
    print(classification_report(y_test, y_pred_kmeans_mapped))
    kmeans_accuracy = accuracy_score(y_test, y_pred_kmeans_mapped)
    print(f"KMeans准确率: {kmeans_accuracy:.2f}")
    print("绘制KMeans混淆矩阵...")
    unique_labels = sorted(list(set(labels)))
    plot_confusion_matrix(y_test, y_pred_kmeans_mapped, labels=unique_labels)
    print("绘制KMeans聚类结果的散点图...")
    plot_kmeans_clusters(images, n_clusters=5)

if __name__ == "__main__":
    main()


