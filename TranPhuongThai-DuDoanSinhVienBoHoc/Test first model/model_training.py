import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Đọc dữ liệu từ file CSV
data = pd.read_csv('data/student_dropout_data.csv')

# Xác định đặc trưng và nhãn
X = data[['age', 'gpa', 'attendance_rate', 'study_hours', 'failed_exams']]
y = data['dropout']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo và huấn luyện mô hình RandomForest
model1 = RandomForestClassifier(n_estimators=100, random_state=42)
model1.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model1.predict(X_test)

# Tính toán độ chính xác
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Lưu mô hình vào file .pkl
with open('model.pkl', 'wb') as file:
    pickle.dump(model1, file)
print("Mô hình đã được lưu vào file 'model.pkl'")
