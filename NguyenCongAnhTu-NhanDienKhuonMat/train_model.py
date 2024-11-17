import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Các thông số huấn luyện
INIT_LR = 1e-4
EPOCHS = 30
BS = 32

# Đường dẫn đến thư mục ảnh và annotations
IMAGE_DIR = "dataset/images"
ANNOTATION_DIR = "dataset/annotations"

# Danh mục nhãn
CATEGORIES = {"with_mask": 0, "without_mask": 1, "mask_weared_incorrect": 2}

data = []
labels = []

# Hàm lấy annotations từ file
def extract_annotation(annotation_file):
    tree = ET.parse(annotation_file)
    root = tree.getroot()
    
    bboxes = []
    label = None
    
    for obj in root.findall('object'):
        label_name = obj.find('name').text
        label = CATEGORIES[label_name]
        bbox = obj.find('bndbox')
        x_min = int(bbox.find('xmin').text)
        y_min = int(bbox.find('ymin').text)
        x_max = int(bbox.find('xmax').text)
        y_max = int(bbox.find('ymax').text)
        
        bboxes.append((x_min, y_min, x_max, y_max))
    
    return bboxes, label

# Duyệt qua từng ảnh và lưu trữ dữ liệu
for image_file in os.listdir(IMAGE_DIR):
    if image_file.endswith(".png"):
        img_path = os.path.join(IMAGE_DIR, image_file)
        annotation_path = os.path.join(ANNOTATION_DIR, image_file.replace(".png", ".xml"))
        image = cv2.imread(img_path)
        bboxes, label = extract_annotation(annotation_path)
        for (x_min, y_min, x_max, y_max) in bboxes:
            face = image[y_min:y_max, x_min:x_max]
            face = cv2.resize(face, (224, 224))
            face = face.astype("float32") / 255.0
            data.append(face)
            labels.append(label)

# Chuyển đổi dữ liệu và nhãn sang mảng numpy
data = np.array(data, dtype="float32")
labels = np.array(labels)
labels = to_categorical(labels, num_classes=3)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.5, stratify=labels, random_state=42)

# Khởi tạo Data Augmentation
aug = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest")

# Xây dựng model dựa trên MobileNetV2
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(3, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)
for layer in baseModel.layers:
    layer.trainable = False
    
# Biên dịch model
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Huấn luyện model
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)

# Lưu model
model.save("mask_detector.h5")

# Dự đoán trên tập kiểm tra
predictions = model.predict(testX)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(testY, axis=1)

import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np

# 1. Vẽ heatmap của confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CATEGORIES.keys(), yticklabels=CATEGORIES.keys())
plt.title("Heatmap của Confusion Matrix")
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
plt.show()

# 2. In confusion matrix và báo cáo phân loại (classification report)
print("Confusion Matrix:")
print(cm)
print("Báo Cáo Phân Loại:")
print(classification_report(true_labels, predicted_labels, target_names=CATEGORIES.keys()))

# 3. Biểu đồ cột (Độ chính xác theo từng lớp)
accuracies = cm.diagonal() / cm.sum(axis=1)
plt.bar(CATEGORIES.keys(), accuracies)
plt.title("Độ Chính Xác Theo Từng Lớp")
plt.xlabel("Lớp")
plt.ylabel("Độ Chính Xác")
plt.show()

# 4. Biểu đồ tròn (Phân phối các lớp trong tập kiểm tra)
class_counts = np.bincount(true_labels)
plt.pie(class_counts, labels=CATEGORIES.keys(), autopct='%1.1f%%', startangle=90)
plt.title("Phân Phối Các Lớp Trong Tập Kiểm Tra")
plt.show()

# 5. Biểu đồ so sánh độ chính xác và mất mát qua các epoch
plt.figure(figsize=(12, 6))

# Độ chính xác
plt.subplot(1, 2, 1)
plt.plot(H.history["accuracy"], label="Độ Chính Xác Huấn Luyện")
plt.plot(H.history["val_accuracy"], label="Độ Chính Xác Kiểm Tra")
plt.title("Độ Chính Xác Qua Các Epoch")
plt.xlabel("Epoch")
plt.ylabel("Độ Chính Xác")
plt.legend()

# Mất mát
plt.subplot(1, 2, 2)
plt.plot(H.history["loss"], label="Mất Mát Huấn Luyện")
plt.plot(H.history["val_loss"], label="Mất Mát Kiểm Tra")
plt.title("Mất Mát Qua Các Epoch")
plt.xlabel("Epoch")
plt.ylabel("Mất Mát")
plt.legend()

plt.tight_layout()
plt.show()

