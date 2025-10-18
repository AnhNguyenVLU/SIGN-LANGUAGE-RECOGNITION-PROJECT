# SIGN-LANGUAGE-RECOGNITION-PROJECT

## Mở đầu 

- Dự án này sử dụng Ultralytics YOLO để thực hiện nhận dạng ngôn ngữ ký hiệu (Sign Language Recognition) theo thời gian thực qua camera.

- Phát hiện và phân loại các ký hiệu tay A–Z từ luồng video trực tiếp.

- Ứng dụng kỹ thuật làm mượt tạm thời (Temporal Smoothing) và ghép ký tự ổn định (Character Aggregation) để hình thành từ một cách chính xác, hạn chế nhiễu và nhấp nháy.

- Tự động chọn ký tự có độ tin cậy (Confidence) cao nhất trong mỗi khung hình và kết hợp theo thời gian để tạo chuỗi từ.

- Cung cấp hiển thị thời gian thực gồm: ký tự hiện tại, ký tự đã làm mượt, FPS, khung phát hiện (Bounding Box) và chuỗi ký tự đang được ghép.

- Hỗ trợ tương tác bàn phím để xóa, thêm khoảng trắng, làm mới hoặc lưu kết quả nhận dạng.

- Tự động ghi kết quả nhận dạng vào tệp output.txt phục vụ cho việc lưu trữ, huấn luyện hoặc phân tích sau này.

## Tổng quan dự án 

| **Thành phần** | **Mô tả** | **Đầu ra** |  
|------------------|--------------------|----------------|  
| **YOLO Detector** | Mô hình **Ultralytics YOLO** nhận dạng ký hiệu tay (A–Z) theo thời gian thực từ camera. | Ký tự được phát hiện cùng độ tin cậy (confidence) và khung phát hiện (bounding box). |  
| **TemporalSmoother** | Làm mượt nhãn theo **cửa sổ trượt** bằng thuật toán **Majority Vote + Confidence Threshold** để giảm nhiễu giữa các khung hình. | Nhãn ký tự ổn định hoặc “BLANK” nếu chưa đủ tin cậy. |  
| **CharAggregator** | Quy tắc **ghép ký tự theo thời gian**: chỉ chốt khi nhãn ổn định liên tiếp K khung hình, tự chèn khoảng trắng nếu im lặng đủ lâu. | Chuỗi ký tự ổn định mô phỏng từ/câu được tạo từ các ký hiệu tay. |  
| **HUD Display** | Hiển thị trực tiếp thông tin nhận dạng lên video gồm: ký tự hiện tại, ký tự đã làm mượt, FPS và chuỗi từ đang hình thành. | Giao diện trực quan thời gian thực trên màn hình video. |  
| **Keyboard Control** | Hỗ trợ phím tắt: **Backspace**, **Space**, **Enter**, **C**, **Q/ESC** để xóa, thêm dấu cách, lưu hoặc thoát chương trình. | Tương tác trực tiếp qua bàn phím khi chạy video. |  
| **Output Writer** | Lưu kết quả nhận dạng cuối cùng vào tệp **`output.txt`** để phục vụ huấn luyện hoặc phân tích sau này. | Chuỗi ký tự hoàn chỉnh được ghi ra file văn bản. |  

## Công nghệ sử dụng

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — Mô hình **phát hiện ký hiệu tay (A–Z)** theo thời gian thực, dùng để phân loại cử chỉ ngôn ngữ ký hiệu.
  
- [OpenCV](https://opencv.org/) — Thư viện **xử lý ảnh/video**, dùng để truy cập camera, vẽ bounding box, hiển thị HUD và quản lý khung hình theo thời gian thực.
  
- [NumPy](https://numpy.org/) — Hỗ trợ **tính toán ma trận và mảng số học**, phục vụ cho xử lý đầu ra từ YOLO.
  
- [Argparse](https://docs.python.org/3/library/argparse.html) — Dùng để **xử lý tham số dòng lệnh**, hỗ trợ truyền các tùy chọn như model path, camera index, threshold,…
  
- [Collections (Deque, Counter)](https://docs.python.org/3/library/collections.html) — Dùng cho **làm mượt thời gian** và đếm nhãn xuất hiện theo cửa sổ trượt.
   
- [Typing](https://docs.python.org/3/library/typing.html) — Sử dụng **type hints** để tăng khả năng đọc và bảo trì mã nguồn.

- [Time / Datetime](https://docs.python.org/3/library/time.html) — Dùng để **đồng bộ thời gian thực** khi xử lý khung hình và chèn dấu cách giữa các ký tự.
  
- Python 3.8 – 3.11 — Phiên bản tương thích khuyến nghị cho môi trường chạy.

## Tương tác của người dùng 
  
| **Hành động / Phím tắt** | **Mô tả** |
|----------------------------|------------|
| **Giữ chuột trái + kéo** | Chỉnh vị trí vùng hoặc vạch kiểm tra trực tiếp trên video (nếu mở rộng thêm tính năng). |
| **BACKSPACE** | Xóa ký tự cuối cùng trong chuỗi hiện tại. |
| **SPACE** | Thêm một khoảng trắng vào chuỗi ký tự. |
| **ENTER** | In kết quả hiện tại ra console và lưu vào tệp **`output.txt`**. |
| **C** | Xóa toàn bộ chuỗi ký tự đã ghi. |
| **Q / ESC** | Thoát chương trình một cách an toàn. |

## Chạy code

1. Pip install - r requirements.txt
2.  Python Main.py

## Cấu trúc dữ liệu đầu ra:

| **Thành phần** | **Mô tả** |
|-----------------|------------|
| **output.txt** | Lưu toàn bộ chuỗi ký tự hoặc câu được tạo từ các ký hiệu tay. |
| **Live Display (HUD)** | Hiển thị thông tin thời gian thực: ký tự hiện tại, ký tự làm mượt, FPS và chuỗi đang hình thành. |
| **Console Output** | In ra chuỗi kết quả mỗi khi nhấn **ENTER**, đồng thời xác nhận đã lưu vào file. |

<img width="1305" height="726" alt="image" src="https://github.com/user-attachments/assets/36cf0179-188a-406f-8b30-76c742868e16" />

---
