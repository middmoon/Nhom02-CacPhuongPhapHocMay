    Bước 1: Tạo và chạy máy ảo
    python -m venv venv
    venv\Scripts\activate

    Bước 2: Tải thư viện
    pip install tensorflow opencv-python numpy scikit-learn matplotlib

    Bước 3: Train Model
    python train_model.py ( Lưu ý nếu đã có file mask_detector.h5 thì bỏ qua bước này nếu không chạy được model hãy xóa đi và train lại)

    Bước 4: Chạy chương trình
    python main.py 