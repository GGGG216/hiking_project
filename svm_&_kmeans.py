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

# 定义数据加载函数
def load_data(image_folder, image_size=(512,512)):
    images = []
    labels = []
    for file in os.listdir(image_folder):
        if file.endswith('.npy'):  # 检查是否为图片文件
            # 加载图片
            npy_path = os.path.join(image_folder, file)
            image = np.load(npy_path)

            # 加载对应的标签
            label_file = file.replace('.npy', '.txt')
            label_path = os.path.join(image_folder, label_file)
            if os.path.exists(label_path):  # 检查标签文件是否存在
                with open(label_path, 'r') as f:
                    label = int(f.read().strip())
                    images.append(image)
                    labels.append(label)
            # Note: This reshape operation seems incorrect and unnecessary here
            # image.reshape(-1, 2)  # This doesn't modify the image in place
    return np.array(images), np.array(labels)

# 绘制混淆矩阵
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

# 绘制KMeans聚类结果的散点图
def plot_kmeans_clusters(images, n_clusters=5):
    # Step 1: Flatten the images to 2D (n_samples, n_features)
    # Assuming images shape is (n_samples, height, width) or (n_samples, height, width, channels)
    n_samples = images.shape[0]
    flattened_images = images.reshape(n_samples, -1)  # Flatten each image into a 1D array

    # Step 2: Apply KMeans clustering on the flattened data
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(flattened_images)

    # Step 3: Reduce dimensionality to 2D using PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    reduced_features = pca.fit_transform(flattened_images)

    # Step 4: Plot the 2D features with clusters
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=clusters, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Cluster')
    plt.title("KMeans Clustering Results (PCA Reduced)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()

# 主函数
def main():
    # 数据集路径
    image_folder = './data/depth_maps/train'  # 替换为你的数据集路径

    # 加载数据
    print("加载数据...")
    images, labels = load_data(image_folder)
    print(f"加载图片数量: {len(images)}, 标签数量: {len(labels)}")

    # 数据集划分
    print("划分数据集...")
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # 使用KMeans进行聚类预测
    print("使用KMeans进行预测...")
    kmeans = KMeans(n_clusters=5, init="k-means++", random_state=42)
    # Flatten images for KMeans
    n_samples = X_train.shape[0]
    X_train_flattened = X_train.reshape(n_samples, -1)
    kmeans.fit(X_train_flattened)
    # Flatten test images for prediction
    X_test_flattened = X_test.reshape(X_test.shape[0], -1)
    y_pred_kmeans = kmeans.predict(X_test_flattened)

    # 由于KMeans预测的是簇而非标签，需要进行映射
    cluster_to_label = {}
    for cluster in range(kmeans.n_clusters):
        cluster_indices = np.where(kmeans.labels_ == cluster)[0]
        cluster_labels = y_train[cluster_indices]
        if len(cluster_labels) > 0:
            cluster_to_label[cluster] = np.bincount(cluster_labels).argmax()

    y_pred_kmeans_mapped = np.array([cluster_to_label[cluster] for cluster in y_pred_kmeans])

    # 输出KMeans分类报告
    print("KMeans分类报告:")
    print(classification_report(y_test, y_pred_kmeans_mapped))
    kmeans_accuracy = accuracy_score(y_test, y_pred_kmeans_mapped)
    print(f"KMeans准确率: {kmeans_accuracy:.2f}")

    # 绘制KMeans混淆矩阵
    print("绘制KMeans混淆矩阵...")
    unique_labels = sorted(list(set(labels)))
    plot_confusion_matrix(y_test, y_pred_kmeans_mapped, labels=unique_labels)

    # 绘制KMeans聚类结果散点图
    print("绘制KMeans聚类结果的散点图...")
    plot_kmeans_clusters(images, n_clusters=5)

if __name__ == "__main__":
    main()


# # 主函数
# def main():
#     # 数据集路径
#     image_folder = './data/depth_maps/train'  # 替换为你的数据集路径

#     # 加载数据
#     print("加载数据...")
#     images, labels = load_data(image_folder)
#     print(f"加载图片数量: {len(images)}, 标签数量: {len(labels)}")

   


#     # # 数据集划分
#     # print("划分数据集...")
#     X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, random_state=42)

#     # # 使用SVM训练模型
#     # print("训练SVM模型...")
#     # svm = SVC(kernel='linear')
#     # svm.fit(X_train, y_train)

#     # # 测试SVM模型
#     # print("测试SVM模型...")
#     # y_pred_svm = svm.predict(X_test)

#     # # 输出SVM分类报告
#     # print("SVM分类报告:")
#     # print(classification_report(y_test, y_pred_svm))
#     # svm_accuracy = accuracy_score(y_test, y_pred_svm)
#     # print(f"SVM准确率: {svm_accuracy:.2f}")

#     # 使用KMeans进行聚类预测
#     print("使用KMeans进行预测...")
#     kmeans = KMeans(n_clusters=5, init="k-means++",random_state=42)
#     kmeans.fit(images)
#     y_pred_kmeans = kmeans.predict(X_test)

#     # 由于KMeans预测的是簇而非标签，需要进行映射
#     cluster_to_label = {}
#     for cluster in range(kmeans.n_clusters):
#         cluster_indices = np.where(kmeans.labels_ == cluster)[0]
#         cluster_labels = labels[cluster_indices]
#         if len(cluster_labels) > 0:
#             cluster_to_label[cluster] = np.bincount(cluster_labels).argmax()

#     y_pred_kmeans_mapped = np.array([cluster_to_label[cluster] for cluster in y_pred_kmeans])

#     # 输出KMeans分类报告
#     print("KMeans分类报告:")
#     print(classification_report(y_test, y_pred_kmeans_mapped))
#     kmeans_accuracy = accuracy_score(y_test, y_pred_kmeans_mapped)
#     print(f"KMeans准确率: {kmeans_accuracy:.2f}")

#     # # 绘制混淆矩阵对比
#     # print("绘制SVM混淆矩阵...")
#     # unique_labels = sorted(list(set(labels)))
#     # plot_confusion_matrix(y_test, y_pred_svm, labels=unique_labels)

#     print("绘制KMeans混淆矩阵...")
#     plot_confusion_matrix(y_test, y_pred_kmeans_mapped, labels=unique_labels)

#     # 绘制KMeans聚类结果散点图
#     print("绘制KMeans聚类结果的散点图...")
#     plot_kmeans_clusters(images, n_clusters=5)


# if __name__ == "__main__":
#     main()