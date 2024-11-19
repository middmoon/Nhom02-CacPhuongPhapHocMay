NguyenXuanTuongAn-NhanDienChuSoVietTay
- Chạy train.py đề tạo ra model
- Chạy main.py để chạy chương trình


TranPhuongThai-DuDoanSinhVienNghiHoc
- chạy model.ipynb
- chạy app1.py


NguyenMinhThanh-DuBaoThoiTiet
- Tải model về và giải nén (1.7GB) cho vào thư muc NguyenMinhThanh-DuBaoThoiTiet: https://drive.google.com/file/d/1SwdtNh2MFh1TO8wcKDewr4d-ILnhIjEc/view?usp=sharing
- Cài đặt các thư viện Flask và CORS
- chạy cmd: python app.py
- api: File api.http
POST http://127.0.0.1:5011/predict
Content-Type: application/json

{
  "province": "Ho Chi Minh City", 
  "date": "2023-02-20"
}

NguyenCongAnhTu-NhanDienDeoKhauTrang

Bước 1: Tạo và chạy máy ảo
python -m venv venv
venv\Scripts\activate

Bước 2: Tải thư viện
pip install tensorflow opencv-python numpy scikit-learn matplotlib

Bước 3: Train Model
python train_model.py ( Lưu ý nếu đã có file mask_detector.h5 thì bỏ qua bước này nếu không chạy được model hãy xóa đi và train lại)

Bước 4: Chạy chương trình
python main.py 
