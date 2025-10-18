#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chương trình demo: Đọc ký tự ngôn ngữ ký hiệu theo thời gian thực bằng YOLO
và ghép thành từ ổn định (temporal smoothing + quy tắc chốt ký tự).

Yêu cầu cài đặt:
    pip install ultralytics opencv-python numpy

Chạy:
    python sign_reader_yolo.py --model path/to/best.pt

Phím tắt trong cửa sổ video:
    - BACKSPACE  : xóa 1 ký tự cuối
    - SPACE      : thêm khoảng trắng
    - ENTER      : in ra kết quả hiện tại (console) và lưu vào output.txt
    - C         : xóa toàn bộ chuỗi
    - Q/ESC     : thoát

Ghi chú:
    - Mặc định coi "không phát hiện" là BLANK. Nếu mô hình của bạn có
      lớp "blank"/"nothing", có thể truyền --class-blank tương ứng.
    - Script chọn detection có độ tự tin cao nhất trong nhóm lớp chữ (A-Z).
      Bạn có thể tùy biến bộ lọc lớp bằng --letters "A,B,C,..." nếu dataset khác chuẩn.
"""

import argparse
import time
from collections import deque, Counter
from typing import List, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO

# ============================
# Các lớp xử lý thời gian
# ============================
class TemporalSmoother:
    """Làm mượt nhãn theo cửa sổ trượt bằng majority vote + ngưỡng confidence.
    Trả về nhãn đã làm mượt (hoặc 'BLANK' nếu chưa đủ chắc chắn).
    """
    def __init__(self, window: int = 10, p_major: float = 0.6, conf_th: float = 0.7):
        self.window = window
        self.p_major = p_major
        self.conf_th = conf_th
        self.buf = deque(maxlen=window)  # (label, conf)

    def update(self, label: str, conf: float) -> Tuple[str, float]:
        self.buf.append((label, conf))
        if len(self.buf) == 0:
            return 'BLANK', 0.0
        labels = [x[0] for x in self.buf]
        confs  = [x[1] for x in self.buf]
        lab, cnt = Counter(labels).most_common(1)[0]
        conf_avg = float(sum(confs) / len(confs)) if confs else 0.0
        ok = (cnt / len(labels) >= self.p_major) and (conf_avg >= self.conf_th)
        return (lab if ok and lab != 'BLANK' else 'BLANK', conf_avg)

class CharAggregator:
    """Quy tắc chốt ký tự & ghép chuỗi có chống lặp và chèn dấu cách theo khoảng BLANK.
    - Chỉ chốt khi nhãn ổn định liên tiếp K frame (K-stable)
    - Chống trùng: không lặp cùng ký tự nếu chưa có đủ BLANK ở giữa
    - Space: nếu BLANK kéo dài >= space_ms thì chèn khoảng trắng
    """
    def __init__(self, k_stable: int = 4, gap_blank_frames: int = 8, space_ms: int = 800):
        self.k_stable = k_stable
        self.gap_blank_frames = gap_blank_frames
        self.space_ms = space_ms

        self.recent = deque(maxlen=k_stable)
        self.blank_count = 0
        self.last_nonblank_ms = int(time.time() * 1000)
        self.last_appended: Optional[str] = None
        self.text: List[str] = []

    def push(self, label: str, t_ms: int) -> Optional[str]:
        # đếm BLANK & cập nhật thời điểm thấy ký tự gần nhất
        if label == 'BLANK':
            self.blank_count += 1
        else:
            self.blank_count = 0
            self.last_nonblank_ms = t_ms

        # Lưu vào cửa sổ ổn định
        self.recent.append(label)

        # Kiểm tra ổn định K frame
        appended = None
        if len(self.recent) == self.k_stable and len(set(self.recent)) == 1:
            stable_label = self.recent[0]
            if stable_label != 'BLANK':
                can_repeat = (self.last_appended != stable_label) or (self.blank_count >= self.gap_blank_frames)
                if can_repeat:
                    self.text.append(stable_label)
                    self.last_appended = stable_label
                    appended = stable_label

        # Chèn khoảng trắng nếu im lặng đủ lâu
        if (t_ms - self.last_nonblank_ms) >= self.space_ms:
            if self.text and self.text[-1] != ' ':
                self.text.append(' ')
                appended = ' '
        return appended

    # Các tiện ích chỉnh sửa bằng phím tắt
    def backspace(self):
        if self.text:
            self.text.pop()

    def clear(self):
        self.text = []

    def get_text(self) -> str:
        return ''.join(self.text).strip()

# ============================
# YOLO & tiện ích
# ============================

def build_letter_set(model_names: dict, explicit_letters: Optional[str]) -> set:
    """Xác định tập lớp chữ hợp lệ.
    - Nếu user truyền --letters: dùng danh sách đó (phân tách bằng dấu phẩy)
    - Ngược lại: lấy những lớp 1 ký tự alphabet (A-Z/a-z) từ model.names
    """
    if explicit_letters:
        raw = [s.strip() for s in explicit_letters.split(',') if s.strip()]
        return set(raw)
    letters = set()
    for _, name in model_names.items():
        if len(name) == 1 and name.isalpha():
            letters.add(name.upper())
    # fallback: nếu model đặt tên kiểu 'A_sign', 'B_sign'...
    if not letters:
        for _, name in model_names.items():
            short = name.split('_')[0]
            if len(short) == 1 and short.isalpha():
                letters.add(short.upper())
    return letters


def top_detection_as_letter(results, letters: set, class_blank: Optional[str], conf_min: float) -> Tuple[str, float, Optional[Tuple[int,int,int,int]], Optional[str], float]:
    """Chọn detection có confidence cao nhất trong nhóm lớp chữ.
    Trả về (label|BLANK, conf, bbox(x1,y1,x2,y2) hoặc None, raw_class_name hoặc None, raw_conf)
    """
    if results is None or len(results.boxes) == 0:
        return 'BLANK', 0.0, None, None, 0.0

    boxes = results.boxes
    cls = boxes.cls.detach().cpu().numpy().astype(int)
    conf = boxes.conf.detach().cpu().numpy()
    xyxy = boxes.xyxy.detach().cpu().numpy().astype(int)

    best_letter = None
    best_letter_conf = 0.0
    best_letter_box = None

    # Nếu có class BLANK rõ ràng
    best_blank_conf = -1.0

    for i in range(len(boxes)):
        name = results.names[int(cls[i])] if hasattr(results, 'names') else None
        score = float(conf[i])
        if name is None:
            continue
        # Ưu tiên chữ cái
        if name.upper() in letters and score >= best_letter_conf:
            best_letter_conf = score
            best_letter = name.upper()
            best_letter_box = tuple(map(int, xyxy[i]))
        # Ghi nhận blank nếu được chỉ định
        if class_blank and name.lower() == class_blank.lower():
            if score > best_blank_conf:
                best_blank_conf = score

    # Quyết định đầu ra
    if best_letter is not None and best_letter_conf >= conf_min:
        return best_letter, best_letter_conf, best_letter_box, best_letter, best_letter_conf

    # Nếu có dự đoán blank đáng tin
    if class_blank and best_blank_conf >= conf_min:
        return 'BLANK', best_blank_conf, None, class_blank, best_blank_conf

    # Mặc định coi như BLANK
    return 'BLANK', 0.0, None, None, 0.0


def draw_hud(frame, text_line: str, curr_label: str, smooth_label: str, fps: float,
             box: Optional[Tuple[int,int,int,int]] = None):
    h, w = frame.shape[:2]
    # BBox
    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # Thanh nền dưới
    cv2.rectangle(frame, (0, h - 90), (w, h), (0, 0, 0), -1)
    # Hiển thị thông tin
    cv2.putText(frame, f"Live: {curr_label}", (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Smooth: {smooth_label}", (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 130, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 200, 255), 2)
    # Dòng text đang ghép
    cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.putText(frame, text_line, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


def main():
    parser = argparse.ArgumentParser(description='ASL/Sign letter to word with YOLO + temporal smoothing')
    parser.add_argument('--model', type=str, required=True, help='Đường dẫn .pt YOLO (ultralytics)')
    parser.add_argument('--camera', type=int, default=0, help='Chỉ số camera (mặc định 0)')
    parser.add_argument('--api', type=str, default='auto', choices=['auto', 'dshow', 'msmf'], help='Chọn backend camera: auto/dshow/msmf')
    parser.add_argument('--conf', type=float, default=0.35, help='Ngưỡng confidence tối thiểu để nhận ký tự')
    parser.add_argument('--window', type=int, default=10, help='Cửa sổ làm mượt (frame)')
    parser.add_argument('--kstable', type=int, default=4, help='Số frame ổn định để chốt ký tự')
    parser.add_argument('--pmajor', type=float, default=0.6, help='Tỉ lệ majority tối thiểu trong cửa sổ')
    parser.add_argument('--confth', type=float, default=0.7, help='Ngưỡng trung bình confidence trong cửa sổ')
    parser.add_argument('--gap', type=int, default=8, help='Số frame BLANK tối thiểu để cho phép lặp ký tự')
    parser.add_argument('--space_ms', type=int, default=800, help='Thời gian BLANK để chèn khoảng trắng (ms)')
    parser.add_argument('--class-blank', type=str, default=None, help='Tên lớp đại diện BLANK (ví dụ: blank hoặc nothing)')
    parser.add_argument('--letters', type=str, default=None, help='Danh sách lớp chữ, ví dụ: "A,B,C,...". Bỏ qua để tự suy luận từ model.names')
    parser.add_argument('--width', type=int, default=960, help='Resize chiều rộng hiển thị (0 = giữ nguyên)')
    # Camera tuning (giải quyết lỗi MSMF không grab được frame):
    parser.add_argument('--cam-width', type=int, default=1280, help='Chiều rộng camera yêu cầu (ví dụ 640/1280)')
    parser.add_argument('--cam-height', type=int, default=720, help='Chiều cao camera yêu cầu (ví dụ 480/720)')
    parser.add_argument('--fps', type=int, default=30, help='FPS yêu cầu từ camera')
    parser.add_argument('--fourcc', type=str, default='MJPG', help='FOURCC ưu tiên (ví dụ MJPG, YUY2, H264, RAW). Để trống nếu không muốn đặt')
    args = parser.parse_args()

    def open_camera(index: int, api: str):
        # Thử mở theo ưu tiên để tránh lỗi MSMF (-1072875772)
        backends = []
        if api == 'msmf':
            backends = [cv2.CAP_MSMF]
        elif api == 'dshow':
            backends = [cv2.CAP_DSHOW]
        else:
            backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, 0]  # 0 = OpenCV tự chọn

        last_err = None
        for be in backends:
            cap = cv2.VideoCapture(index, be) if isinstance(be, int) and be != 0 else cv2.VideoCapture(index)
            if be == cv2.CAP_MSMF:
                be_name = 'MSMF'
            elif be == cv2.CAP_DSHOW:
                be_name = 'DSHOW'
            else:
                be_name = 'AUTO'

            if not cap.isOpened():
                last_err = f'Không mở được camera với backend {be_name}'
                continue

            # Set tham số an toàn cho Windows
            if args.fourcc:
                try:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
                except Exception:
                    pass
            if args.cam_width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
            if args.cam_height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
            if args.fps:
                cap.set(cv2.CAP_PROP_FPS, args.fps)

            # Warmup & kiểm tra grab
            ok, _ = cap.read()
            if not ok:
                cap.release()
                last_err = f'Grab frame thất bại với backend {be_name}'
                continue

            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"[INFO] Camera opened with backend {be_name} at {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")
            return cap, be_name

        raise RuntimeError(last_err or 'Không thể mở camera với bất kỳ backend nào')

    # Tải model
    print('[INFO] Loading model...')
    model = YOLO(args.model)

    # Xác định tập chữ cái
    model_names = getattr(model, 'names', None)
    if model_names is None:
        model_names = getattr(getattr(model, 'model', None), 'names', {}) or {}
    letters = build_letter_set(model_names, args.letters)
    if not letters:
        print('[WARN] Không suy luận được tập chữ từ model.names, vui lòng dùng --letters "A,B,C,..."')
    else:
        print('[INFO] LETTERS =', sorted(list(letters)))
    if args.class_blank:
        print('[INFO] Using explicit BLANK class =', args.class_blank)

    # Khởi tạo camera với fallback backend
    cap, backend_name = open_camera(args.camera, args.api)

    # FPS đo bằng EMA để ổn định hiển thị
    fps, alpha = 0.0, 0.9

    smoother = TemporalSmoother(window=args.window, p_major=args.pmajor, conf_th=args.confth)
    aggregator = CharAggregator(k_stable=args.kstable, gap_blank_frames=args.gap, space_ms=args.space_ms)

    print('[INFO] Bắt đầu. Nhấn Q hoặc ESC để thoát.')

    try:
        failed_grabs = 0
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                failed_grabs += 1
                if failed_grabs > 30:
                    raise RuntimeError('Camera không cung cấp frame (kiểm tra quyền truy cập, thiết bị bận, hoặc thử --api dshow).')
                continue
            failed_grabs = 0

            # Suy luận YOLO
            results_list = model(frame, verbose=False)
            results = results_list[0]

            label, conf, box, raw_name, raw_conf = top_detection_as_letter(
                results, letters, args.class_blank, args.conf)

            # Làm mượt
            smooth_label, _ = smoother.update(label, conf)
            now_ms = int(time.time() * 1000)

            # Ghép ký tự
            _ = aggregator.push(smooth_label if smooth_label != 'BLANK' else 'BLANK', now_ms)

            # HUD
            text_line = aggregator.get_text()

            # Resize hiển thị (không ảnh hưởng suy luận)
            draw_frame = frame
            box_scaled = box
            if args.width > 0:
                h, w = frame.shape[:2]
                scale = args.width / float(w)
                draw_frame = cv2.resize(frame, (args.width, int(h * scale)))
                if box is not None:
                    x1, y1, x2, y2 = box
                    x1 = int(x1 * scale); y1 = int(y1 * scale)
                    x2 = int(x2 * scale); y2 = int(y2 * scale)
                    box_scaled = (x1, y1, x2, y2)

            # cập nhật FPS
            dt = time.time() - t0
            inst_fps = 1.0 / dt if dt > 0 else 0.0
            fps = alpha * fps + (1 - alpha) * inst_fps if fps > 0 else inst_fps

            draw_hud(draw_frame, text_line, label if label != 'BLANK' else 'BLANK', smooth_label, fps, box_scaled)

            if box_scaled is not None and raw_name is not None:
                x1, y1, x2, y2 = box_scaled
                cv2.putText(draw_frame, f"{raw_name}:{raw_conf:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow('Sign Reader (YOLO + Temporal Aggregation)', draw_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == 8:  # Backspace
                aggregator.backspace()
            elif key == 32:  # Space
                s = aggregator.get_text()
                if not s.endswith(' '):
                    aggregator.text.append(' ')
            elif key in (10, 13):  # Enter
                final_text = aggregator.get_text()
                print('[RESULT]', final_text)
                with open('output.txt', 'a', encoding='utf-8') as f:
                    f.write(final_text + ' ')
                cv2.displayStatusBar('Sign Reader (YOLO + Temporal Aggregation)', f'Đã lưu: {final_text}', 2000)
            elif key in (ord('c'), ord('C')):
                aggregator.clear()

    finally:
        cap.release()
        cv2.destroyAllWindows()



if __name__ == '__main__':
    main()