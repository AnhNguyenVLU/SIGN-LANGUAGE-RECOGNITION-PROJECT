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
