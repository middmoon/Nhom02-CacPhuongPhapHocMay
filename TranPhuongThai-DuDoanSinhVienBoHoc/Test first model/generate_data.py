import pandas as pd
import numpy as np

# Thiết lập seed để đảm bảo dữ liệu ngẫu nhiên có thể tái tạo
np.random.seed(42)

# Số lượng sinh viên
num_students = 1000

# Tạo dữ liệu giả lập
ages = np.random.randint(18, 30, size=num_students)  # Độ tuổi từ 18 đến 30
gpas = np.round(np.random.uniform(1.0, 4.0, size=num_students), 2)  # GPA từ 1.0 đến 4.0
attendance_rates = np.round(np.random.uniform(50, 100, size=num_students), 2)  # Tỉ lệ tham dự lớp từ 50% đến 100%
study_hours = np.round(np.random.uniform(1, 10, size=num_students), 2)  # Giờ học mỗi tuần từ 1 đến 10 giờ
failed_exams = np.random.randint(0, 5, size=num_students)  # Số lượng kỳ thi bị trượt từ 0 đến 5
status = np.random.choice([0, 1], size=num_students, p=[0.7, 0.3])  # 70% không bỏ học, 30% bỏ học

# Tạo DataFrame
df = pd.DataFrame({
    'age': ages,
    'gpa': gpas,
    'attendance_rate': attendance_rates,
    'study_hours': study_hours,
    'failed_exams': failed_exams,
    'dropout': status
})

# Lưu dữ liệu thành file CSV
df.to_csv('data/student_dropout_data.csv', index=False)
print("Dữ liệu đã được lưu vào file 'student_dropout_data.csv'")
