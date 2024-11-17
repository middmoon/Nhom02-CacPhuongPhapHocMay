import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, mean_squared_error
from tensorflow.keras import layers, models, optimizers
from keras.datasets import mnist
from keras.utils import to_categorical
import time

# Load dữ liệu MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print("Kích thước tập huấn luyện:", train_images.shape)
print("Kích thước tập kiểm thử:", test_images.shape)
unique, counts = np.unique(train_labels, return_counts=True)
print("Số lượng ảnh theo từng nhãn trong tập huấn luyện:", dict(zip(unique, counts)))

# Chuẩn bị dữ liệu
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
train_labels_cat = to_categorical(train_labels)
test_labels_cat = to_categorical(test_labels)

# Xây dựng mô hình CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Cấu hình mô hình với learning rate tùy chỉnh
learning_rate = 0.001
model.compile(optimizer=optimizers.RMSprop(learning_rate=learning_rate), 
              loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Huấn luyện mô hình và tính thời gian hoàn thành
start_time = time.time()  # Bắt đầu đo thời gian
history = model.fit(train_images, train_labels_cat, epochs=5, batch_size=64, validation_data=(test_images, test_labels_cat))
end_time = time.time()  # Kết thúc đo thời gian

# Tính thời gian huấn luyện
training_time = end_time - start_time
print(f"Thời gian huấn luyện hoàn thành: {training_time:.2f} giây")

# Đánh giá mô hình
test_loss, test_acc = model.evaluate(test_images, test_labels_cat)
print("Độ chính xác trên tập kiểm thử:", test_acc)

# Dự đoán và tính toán các chỉ số đánh giá khác
y_pred = np.argmax(model.predict(test_images), axis=1)
f1_macro = f1_score(test_labels, y_pred, average='macro')
f1_micro = f1_score(test_labels, y_pred, average='micro')
precision = precision_score(test_labels, y_pred, average='macro')
recall = recall_score(test_labels, y_pred, average='macro')
mse = mean_squared_error(test_labels, y_pred)
rmse = np.sqrt(mse)

print(f"F1 Macro: {f1_macro}")
print(f"F1 Micro: {f1_micro}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")

# Lưu mô hình
model.save('mnist.h5')

# Vẽ biểu đồ Confusion Matrix
cm = confusion_matrix(test_labels, y_pred)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title("Confusion Matrix")
plt.show()

# Biểu đồ mất mát và độ chính xác
epochs = range(1, len(history.history['accuracy']) + 1)
plt.plot(epochs, history.history['accuracy'], 'bo-', label='Độ chính xác huấn luyện')
plt.plot(epochs, history.history['val_accuracy'], 'go-', label='Độ chính xác kiểm thử')
plt.title('Độ chính xác qua từng epoch')
plt.xlabel('Epochs')
plt.ylabel('Độ chính xác')
plt.legend()
plt.show()

plt.plot(epochs, history.history['loss'], 'bo-', label='Mất mát huấn luyện')
plt.plot(epochs, history.history['val_loss'], 'go-', label='Mất mát kiểm thử')
plt.title('Mất mát qua từng epoch')
plt.xlabel('Epochs')
plt.ylabel('Mất mát')
plt.legend()
plt.show()
