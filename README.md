# PPE Detection System

## Giới thiệu

**PPE Detection System** là một hệ thống ứng dụng trí tuệ nhân tạo nhằm tự động phát hiện và giám sát việc tuân thủ quy định về đồ bảo hộ lao động (PPE) tại các công trường, nhà máy, xưởng sản xuất, khu vực xây dựng,... Hệ thống giúp nâng cao ý thức an toàn lao động, giảm thiểu rủi ro tai nạn và hỗ trợ quản lý hiệu quả hơn.

## Tính năng nổi bật

- **Nhận diện người:** Xác định chính xác vị trí người lao động trong khung hình.
- **Phát hiện mũ bảo hộ:** Kiểm tra người lao động có đội mũ bảo hộ đúng quy định hay không.
- **Phát hiện áo phản quang:** Kiểm tra người lao động có mặc áo phản quang đúng quy định hay không.
- **Cảnh báo vi phạm:** Phát âm thanh cảnh báo và ghi log khi phát hiện trường hợp không tuân thủ PPE.
- **Thống kê & báo cáo:** Thống kê số lượng người tuân thủ, vi phạm, tỷ lệ tuân thủ và xuất báo cáo chi tiết.
- **Lưu trữ sự kiện:** Lưu lại hình ảnh, video, log các sự kiện vi phạm để phục vụ kiểm tra, đánh giá sau này.
- **Giao diện trực quan:** Giao diện người dùng thân thiện, dễ sử dụng, hỗ trợ thao tác nhanh chóng.
- **Hỗ trợ nhiều nguồn video:** Webcam, file video, ảnh tĩnh, camera IP (RTSP).
- **Tùy chỉnh độ nhạy:** Cho phép điều chỉnh ngưỡng phát hiện phù hợp với từng môi trường thực tế.

## Công nghệ sử dụng

- **Ngôn ngữ:** Python 3.8+
- **Mô hình AI:** YOLOv8 Pose (Ultralytics)
- **Thư viện:** OpenCV, Numpy, Pillow, Tkinter, Pygame, Ultralytics, v.v.

## Cài đặt & Hướng dẫn sử dụng

### 1. Clone dự án

```bash
git clone https://github.com/NgHaiii/BTL-TGMT.git
cd BTL-TGMT
```

### 2. Cài đặt thư viện cần thiết

1. **Cài đặt PyCharm**  
   Tải và cài đặt từ trang chủ: https://www.jetbrains.com/pycharm/

2. **Cài đặt Anaconda**  
   Link tải trực tiếp:  
   https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Windows-x86_64.exe
3. **Tạo môi trường mới**  
   Mở Anaconda PowerShell Prompt và nhập lệnh:  
   ```
   conda create -n ppe python=3.10
   ```

4. **Kích hoạt môi trường**  
   Nhập lệnh sau vào Anaconda PowerShell Prompt:  
   ```
   conda activate ppe
   ```

5. **Cài đặt các thư viện cần thiết**  
   Nhập các lệnh sau vào terminal (trong PyCharm hoặc Anaconda PowerShell Prompt):
   ```
   conda install numpy pillow pygame
   pip install opencv-python==4.8.1.78 ultralytics pillow
   ```
   *Lưu ý: Một số thư viện như `opencv-python`, `ultralytics` nên cài bằng pip.*

6. **Chọn môi trường Python (Conda) trong PyCharm**
   - Vào menu **File** → **Settings** (`Ctrl+Alt+S`).
   - Chọn **Project: [Tên dự án]** → **Python Interpreter**.
   - Nhấn **Add Interpreter**.
   - Chọn **Conda Environment** → **Select existing**.
   - Ở mục **Path to conda**, chọn đường dẫn tới file `conda.bat` (ví dụ: `C:\Users\PC ASUS\miniconda\condabin\conda.bat`).
   - Ở mục **Environment**, chọn tên môi trường bạn đã tạo (ví dụ: `ppe`).
   - Nhấn **OK** để xác nhận.

Sau khi hoàn tất các bước trên, bạn đã sẵn sàng chạy chương trình trong môi trường đã cấu hình!

### 3. Chuẩn bị mô hình

- Tải file `yolov8n-pose.pt` từ Ultralytics hoặc trang chủ YOLOv8 và đặt vào thư mục dự án.

### 4. Chạy chương trình

```bash
python main.py
```

### 5. Sử dụng giao diện

- **Chọn nguồn video:** Webcam, file video, ảnh hoặc nhập link RTSP camera IP.
- **Chọn các tùy chọn PPE cần kiểm tra:** Nhận diện người, mũ bảo hộ, áo phản quang.
- **Điều chỉnh độ nhạy phát hiện** phù hợp với môi trường thực tế.
- **Nhấn "Bắt đầu"** để hệ thống tiến hành nhận diện và giám sát.
- **Theo dõi kết quả trực tiếp** trên giao diện: khung nhận diện, trạng thái tuân thủ, cảnh báo, thống kê.
- **Xem báo cáo, xuất log, lưu hình ảnh/video** các sự kiện vi phạm.

## Sơ đồ vận hành hệ thống

<img width="1024" height="1536" alt="Copilot_20250724_101040" src="https://github.com/user-attachments/assets/02eef93e-0d49-45af-8691-bc834185a4b4" />

**Mô tả sơ đồ:**
- Camera truyền tín hiệu video qua giao thức RTSP.
- Hệ thống tải mô hình và thực hiện nhận diện người, kiểm tra PPE bằng YOLOv8 Pose.
- Kết quả được phân tích, vẽ lên khung hình, đồng thời xử lý cảnh báo và lưu trữ sự kiện.
- Người dùng có thể xem trực tiếp, xuất báo cáo, log và dữ liệu hình ảnh/video.

## Cấu trúc thư mục dự án

```
BTL-TGMT/
├── .idea/
├── README.md
├── main.py
├── requirements.txt
├── sound.wav
└── yolov8n-pose.pt
```

## Đóng góp & Hỗ trợ

- Nếu bạn phát hiện lỗi hoặc có ý tưởng cải tiến, hãy tạo [Issue](https://github.com/NgHaiii/BTL-TGMT/issues) hoặc gửi Pull Request.
- Mọi thắc mắc vui lòng liên hệ nhóm phát triển qua GitHub hoặc email.

---


