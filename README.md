# SIGN-LANGUAGE-RECOGNITION-PROJECT

## Mở đầu 

- Dự án này sử dụng Ultralytics YOLO để thực hiện nhận dạng ngôn ngữ ký hiệu (Sign Language Recognition) theo thời gian thực qua camera.

- Phát hiện và phân loại các ký hiệu tay A–Z từ luồng video trực tiếp.

- Ứng dụng kỹ thuật làm mượt tạm thời (Temporal Smoothing) và ghép ký tự ổn định (Character Aggregation) để hình thành từ một cách chính xác, hạn chế nhiễu và nhấp nháy.

- Tự động chọn ký tự có độ tin cậy (Confidence) cao nhất trong mỗi khung hình và kết hợp theo thời gian để tạo chuỗi từ.

- Cung cấp hiển thị thời gian thực gồm: ký tự hiện tại, ký tự đã làm mượt, FPS, khung phát hiện (Bounding Box) và chuỗi ký tự đang được ghép.

- Hỗ trợ tương tác bàn phím để xóa, thêm khoảng trắng, làm mới hoặc lưu kết quả nhận dạng.

- Tự động ghi kết quả nhận dạng vào tệp output.txt phục vụ cho việc lưu trữ, huấn luyện hoặc phân tích sau này.
