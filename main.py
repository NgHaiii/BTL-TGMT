import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import time
import os
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
import pygame
from datetime import datetime

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

class PersonTrack:
    def __init__(self, box, helmet, vest):
        self.box = box
        self.helmet = helmet
        self.vest = vest
        self.violation = self._calc_violation()
        self.counted = False

    def update(self, box, helmet, vest):
        changed = (self.helmet != helmet) or (self.vest != vest)
        self.box = box
        self.helmet = helmet
        self.vest = vest
        old_violation = self.violation
        self.violation = self._calc_violation()
        return changed or (old_violation != self.violation)

    def _calc_violation(self):
        helmet = PPEDetectionGUI.ppe_helmet
        vest = PPEDetectionGUI.ppe_vest
        if helmet and not self.helmet:
            return True
        if vest and not self.vest:
            return True
        return False

class PPEDetectionGUI:
    ppe_helmet = True
    ppe_vest = True

    def __init__(self, root):
        self.root = root
        self.root.title("🛡️ PPE Detection System - Hệ thống phát hiện đồ bảo hộ lao động")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        self.video_source = None
        self.cap = None
        self.model = None
        self.is_running = False
        self.is_paused = False
        self.detection_thread = None
        self.stats = {
            'total_detections': 0,
            'total_violations': 0,
            'total_compliant': 0,
            'session_start': None
        }
        self.detection_config = {
            'confidence_threshold': 0.87,
            'enable_sound': True,
            'show_boxes': True,
        }
        self.tracked_persons = []
        self._init_sound()
        self._init_gui()
        self._load_model()

    def _init_sound(self):
        try:
            pygame.mixer.init()
            sample_rate, duration, frequency = 44100, 0.5, 800
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = np.sin(2 * np.pi * frequency * t) * 0.3
            audio = (audio * 32767).astype(np.int16)
            stereo_audio = np.array([audio, audio]).T.copy()
            self.alert_sound = pygame.sndarray.make_sound(stereo_audio)
        except Exception as e:
            print(f"Sound initialization error: {e}")
            self.alert_sound = None

    def _init_gui(self):
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self._create_title(main_container)
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True)
        self._create_control_panel(content_frame)
        self._create_video_panel(content_frame)
        self._create_stats_panel(content_frame)

    def _create_title(self, parent):
        title_frame = ttk.Frame(parent)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(title_frame, text="🛡️ HỆ THỐNG AN TOÀN LAO ĐỘNG",
                  font=("Arial", 18, "bold")).pack()

    def _create_control_panel(self, parent):
        left_panel = ttk.LabelFrame(parent, text="🎛️ Điều khiển", padding=5, width=280)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        canvas = tk.Canvas(left_panel)
        scrollbar = ttk.Scrollbar(left_panel, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>",
                              lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        self._add_source_selection(scrollable_frame)
        self._add_control_buttons(scrollable_frame)
        self._add_ppe_options(scrollable_frame)
        self._add_sensitivity_settings(scrollable_frame)
        self._add_general_settings(scrollable_frame)
        self._add_action_buttons(scrollable_frame)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

    def _add_source_selection(self, parent):
        source_frame = ttk.LabelFrame(parent, text="📹 Nguồn video", padding=5)
        source_frame.pack(fill=tk.X, pady=5)
        self.source_var = tk.StringVar(value="webcam")
        ttk.Radiobutton(source_frame, text="📷 Webcam",
                        variable=self.source_var, value="webcam").pack(anchor=tk.W)
        ttk.Radiobutton(source_frame, text="📁 File Video",
                        variable=self.source_var, value="video").pack(anchor=tk.W)
        ttk.Radiobutton(source_frame, text="🖼️ Ảnh",
                        variable=self.source_var, value="image").pack(anchor=tk.W)
        ttk.Button(source_frame, text="📂 Chọn File",
                   command=self._select_file).pack(fill=tk.X, pady=2)
        self.file_label = ttk.Label(source_frame, text="Chưa chọn file",
                                    foreground="gray", font=("Arial", 8))
        self.file_label.pack(fill=tk.X, pady=2)

    def _add_control_buttons(self, parent):
        control_frame = ttk.LabelFrame(parent, text="🎮 Điều khiển", padding=5)
        control_frame.pack(fill=tk.X, pady=5)
        self.start_btn = ttk.Button(control_frame, text="▶️ Bắt đầu",
                                    command=self._start_detection)
        self.start_btn.pack(fill=tk.X, pady=2)
        self.pause_btn = ttk.Button(control_frame, text="⏸️ Tạm dừng",
                                    command=self._pause_detection, state="disabled")
        self.pause_btn.pack(fill=tk.X, pady=2)
        self.stop_btn = ttk.Button(control_frame, text="⏹️ Dừng",
                                   command=self._stop_detection, state="disabled")
        self.stop_btn.pack(fill=tk.X, pady=2)

    def _add_ppe_options(self, parent):
        ppe_frame = ttk.LabelFrame(parent, text="🛡️ PPE Detection", padding=5)
        ppe_frame.pack(fill=tk.X, pady=5)
        self.person_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ppe_frame, text="👤 Nhận diện người", variable=self.person_var).pack(anchor=tk.W)
        self.helmet_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ppe_frame, text="⛑️ Phát hiện mũ bảo hiểm", variable=self.helmet_var).pack(anchor=tk.W)
        self.vest_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ppe_frame, text="🦺 Phát hiện áo phản quang", variable=self.vest_var).pack(anchor=tk.W)

    def _add_sensitivity_settings(self, parent):
        sensitivity_frame = ttk.LabelFrame(parent, text="🎯 Độ nhạy phát hiện", padding=5)
        sensitivity_frame.pack(fill=tk.X, pady=5)
        self.sensitivity_var = tk.StringVar(value="normal")
        ttk.Radiobutton(sensitivity_frame, text="🔴 Cao (ít bỏ sót)",
                        variable=self.sensitivity_var, value="high",
                        command=self._update_sensitivity).pack(anchor=tk.W)
        ttk.Radiobutton(sensitivity_frame, text="🟡 Bình thường",
                        variable=self.sensitivity_var, value="normal",
                        command=self._update_sensitivity).pack(anchor=tk.W)
        ttk.Radiobutton(sensitivity_frame, text="🟢 Thấp (ít báo sai)",
                        variable=self.sensitivity_var, value="low",
                        command=self._update_sensitivity).pack(anchor=tk.W)

    def _add_general_settings(self, parent):
        general_frame = ttk.LabelFrame(parent, text="🔧 Cài đặt chung", padding=5)
        general_frame.pack(fill=tk.X, pady=5)
        self.sound_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(general_frame, text="🔊 Âm thanh",
                        variable=self.sound_var).pack(anchor=tk.W)
        self.boxes_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(general_frame, text="📦 Hiện khung",
                        variable=self.boxes_var).pack(anchor=tk.W)

    def _add_action_buttons(self, parent):
        action_frame = ttk.LabelFrame(parent, text="📋 Hành động", padding=5)
        action_frame.pack(fill=tk.X, pady=5)
        actions = [
            ("📈 Thống kê tổng quan", self._show_overview_stats),
            ("📊 Báo cáo chi tiết", self._show_report),
            ("🔄 Reset", self._reset_stats)
        ]
        for text, command in actions:
            ttk.Button(action_frame, text=text, command=command).pack(fill=tk.X, pady=1)

    def _create_video_panel(self, parent):
        center_panel = ttk.LabelFrame(parent, text="📺 Video Display", padding=10)
        center_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        self.video_frame = ttk.Frame(center_panel)
        self.video_frame.pack(fill=tk.BOTH, expand=True)
        self.video_label = ttk.Label(self.video_frame,
                                     text="📷 Video sẽ hiển thị ở đây\nNhấn 'Bắt đầu' để khởi động",
                                     font=("Arial", 12), background="black",
                                     foreground="white", anchor="center")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        status_frame = ttk.Frame(center_panel)
        status_frame.pack(fill=tk.X, pady=5)
        ttk.Label(status_frame, text="Trạng thái:").pack(side=tk.LEFT)
        self.status_var = tk.StringVar(value="⏹️ Đã dừng")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var,
                                      foreground="red", font=("Arial", 10, "bold"))
        self.status_label.pack(side=tk.LEFT, padx=5)
        ttk.Label(status_frame, text="Tiến độ:").pack(side=tk.RIGHT, padx=(20, 5))
        self.progress_var = tk.StringVar(value="0/0")
        ttk.Label(status_frame, textvariable=self.progress_var).pack(side=tk.RIGHT)

    def _create_stats_panel(self, parent):
        right_panel = ttk.LabelFrame(parent, text="📊 Thống kê", padding=10, width=350)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        right_panel.pack_propagate(False)
        session_frame = ttk.LabelFrame(right_panel, text="📈 Phiên hiện tại", padding=5)
        session_frame.pack(fill=tk.X, pady=5)
        self.session_time_var = tk.StringVar(value="00:00:00")
        self.detections_var = tk.StringVar(value="0")
        self.compliant_var = tk.StringVar(value="0")
        self.violations_var = tk.StringVar(value="0")
        self.compliance_rate_var = tk.StringVar(value="0%")
        stats_data = [
            ("⏰ Thời gian:", self.session_time_var),
            ("👥 Phát hiện:", self.detections_var),
            ("✅ Tuân thủ:", self.compliant_var, "green"),
            ("⚠️ Vi phạm:", self.violations_var, "red"),
            ("📊 Tỷ lệ tuân thủ:", self.compliance_rate_var, "blue")
        ]
        for i, data in enumerate(stats_data):
            ttk.Label(session_frame, text=data[0]).grid(row=i, column=0, sticky=tk.W)
            color = data[2] if len(data) > 2 else "black"
            ttk.Label(session_frame, textvariable=data[1],
                      foreground=color).grid(row=i, column=1, sticky=tk.W)
        log_frame = ttk.LabelFrame(right_panel, text="📋 Nhật ký trực tiếp", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        text_frame = ttk.Frame(log_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(text_frame, width=35, height=20, wrap=tk.WORD,
                                font=("Consolas", 8))
        log_scrollbar = ttk.Scrollbar(text_frame, orient="vertical",
                                      command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Button(log_frame, text="🗑️ Xóa log",
                   command=self._clear_log).pack(fill=tk.X, pady=2)

    def _load_model(self):
        try:
            self.model = YOLO('yolov8n-pose.pt')
            self._log_message("✅ Model YOLOv8 Pose đã tải thành công")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải model: {e}")
            self._log_message(f"❌ Lỗi tải model: {e}")

    def _select_file(self):
        source_type = self.source_var.get()
        if source_type == "video":
            file_path = filedialog.askopenfilename(
                title="Chọn file video",
                filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                           ("All files", "*.*")]
            )
        elif source_type == "image":
            file_path = filedialog.askopenfilename(
                title="Chọn file ảnh",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                           ("All files", "*.*")]
            )
        else:
            return
        if file_path:
            self.video_source = file_path
            filename = os.path.basename(file_path)
            self.file_label.config(text=filename, foreground="blue")
            self._log_message(f"📁 Đã chọn: {filename}")

    def _update_sensitivity(self):
        sensitivity = self.sensitivity_var.get()
        if sensitivity == "high":
            self.detection_config['confidence_threshold'] = 0.8
        elif sensitivity == "normal":
            self.detection_config['confidence_threshold'] = 0.87
        else:
            self.detection_config['confidence_threshold'] = 0.92
        self._log_message(f"🎯 Đổi độ nhạy: {sensitivity} (conf={self.detection_config['confidence_threshold']:.2f})")

    def _start_detection(self):
        source_type = self.source_var.get()
        if source_type == "webcam":
            self.video_source = 0
        elif source_type in ["video", "image"] and not self.video_source:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn file trước!")
            return
        if source_type == "image":
            self._detect_image()
            return
        self.is_running = True
        self.is_paused = False
        self.stats['session_start'] = datetime.now()
        self.tracked_persons = []
        self._update_status("▶️ Đang chạy", "green")
        self._update_button_states(start=False, pause=True, stop=True)
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()
        self._log_message("🚀 Bắt đầu phát hiện PPE...")

    def _pause_detection(self):
        if self.is_running and not self.is_paused:
            self.is_paused = True
            self._update_status("⏸️ Tạm dừng", "orange")
            self.pause_btn.config(text="▶️ Tiếp tục")
            self._log_message("⏸️ Tạm dừng")
        elif self.is_paused:
            self.is_paused = False
            self._update_status("▶️ Đang chạy", "green")
            self.pause_btn.config(text="⏸️ Tạm dừng")
            self._log_message("▶️ Tiếp tục")

    def _stop_detection(self):
        self.is_running = False
        self.is_paused = False
        if self.cap:
            self.cap.release()
        self._update_status("⏹️ Đã dừng", "red")
        self._update_button_states(start=True, pause=False, stop=False)
        self.video_label.config(image="", text="📷 Video đã dừng")
        self.progress_var.set("0/0")
        self.pause_btn.config(text="⏸️ Tạm dừng")
        self._log_message("⏹️ Đã dừng phát hiện")

    def _detection_loop(self):
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            messagebox.showerror("Lỗi", "Không thể mở nguồn video!")
            self._stop_detection()
            return
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        while self.is_running:
            if self.is_paused:
                time.sleep(0.1)
                continue
            ret, frame = self.cap.read()
            if not ret:
                self._log_message("📺 Video đã kết thúc")
                break
            frame_count += 1
            display_frame = cv2.resize(frame, (640, 480))
            annotated_frame, persons = self._detect_and_draw(display_frame)
            self._update_video_display(annotated_frame)
            if total_frames > 0:
                self.progress_var.set(f"{frame_count}/{total_frames}")
            else:
                self.progress_var.set(f"{frame_count}")
            self._update_session_stats()
            time.sleep(1 / 30)
        self.cap.release()
        self._log_message("🔚 Kết thúc phiên phát hiện")

    def _detect_helmet_by_keypoints(self, frame, kps):
        # Cải thiện: mở rộng vùng kiểm tra, tăng ngưỡng area_ratio, kiểm tra màu sắc kỹ hơn
        head_x, head_y = int(kps[0][0]), int(kps[0][1])
        roi_h = 70
        roi_w = 90
        x1 = max(0, head_x - roi_w // 2)
        x2 = min(frame.shape[1], head_x + roi_w // 2)
        y1 = max(0, head_y - roi_h)
        y2 = head_y
        head_roi = frame[y1:y2, x1:x2]
        if head_roi.size == 0 or head_roi.shape[0] < 10 or head_roi.shape[1] < 10:
            return False
        hsv = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)
        helmet_colors = [
            ([15, 60, 60], [40, 255, 255]),
            ([10, 100, 100], [25, 255, 255]),
            ([0, 100, 100], [10, 255, 255]),
            ([170, 100, 100], [180, 255, 255]),
            ([100, 80, 80], [130, 255, 255]),
            ([45, 80, 80], [75, 255, 255]),
            ([0, 0, 180], [180, 60, 255]),
            ([0, 0, 60], [180, 40, 180]),
        ]
        mask = np.zeros(head_roi.shape[:2], dtype=np.uint8)
        for lower, upper in helmet_colors:
            mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False
        total_area = head_roi.shape[0] * head_roi.shape[1]
        largest = max(contours, key=cv2.contourArea)
        area_ratio = cv2.contourArea(largest) / total_area
        if area_ratio < 0.28:
            return False
        M = cv2.moments(largest)
        if M["m00"] > 0:
            cy = int(M["m01"] / M["m00"])
            if cy > int(head_roi.shape[0] * 0.5):
                return False
        mean_v = np.mean(hsv[..., 2])
        if mean_v < 80:
            return False
        return True

    def _detect_vest_by_keypoints(self, frame, kps):
        # Cải thiện: mở rộng vùng kiểm tra, tăng ngưỡng area_ratio, kiểm tra màu sắc kỹ hơn
        torso_pts = kps[[5, 6, 11, 12]]
        x_min, y_min = np.min(torso_pts, axis=0).astype(int)
        x_max, y_max = np.max(torso_pts, axis=0).astype(int)
        y_mid = (y_min + y_max) // 2
        y1_ext = y_min - int((y_mid - y_min) * 0.5)
        y2_ext = y_max + int((y_max - y_mid) * 0.5)
        x1_ext = x_min - int((x_max - x_min) * 0.18)
        x2_ext = x_max + int((x_max - x_min) * 0.18)
        torso_roi = frame[max(0, y1_ext):min(frame.shape[0], y2_ext), max(0, x1_ext):min(frame.shape[1], x2_ext)]
        if torso_roi.size == 0 or torso_roi.shape[0] < 10 or torso_roi.shape[1] < 10:
            return False
        hsv = cv2.cvtColor(torso_roi, cv2.COLOR_BGR2HSV)
        vest_colors = [
            ([15, 80, 80], [40, 255, 255]),
            ([10, 100, 100], [25, 255, 255]),
            ([0, 100, 100], [10, 255, 255]),
            ([170, 100, 100], [180, 255, 255]),
            ([45, 80, 80], [75, 255, 255]),
            ([0, 0, 180], [180, 60, 255]),
            ([0, 0, 60], [180, 40, 180]),
        ]
        mask = np.zeros(torso_roi.shape[:2], dtype=np.uint8)
        for lower, upper in vest_colors:
            mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False
        total_area = torso_roi.shape[0] * torso_roi.shape[1]
        valid_area = sum(cv2.contourArea(c) for c in contours)
        area_ratio = valid_area / total_area
        if area_ratio < 0.36:
            return False
        mean_v = np.mean(hsv[..., 2])
        if mean_v < 80:
            return False
        return True

    def _detect_and_draw(self, frame):
        if not self.person_var.get():
            return frame, []
        PPEDetectionGUI.ppe_helmet = self.helmet_var.get()
        PPEDetectionGUI.ppe_vest = self.vest_var.get()
        results = self.model(frame, conf=max(self.detection_config['confidence_threshold'], 0.87), verbose=False)
        if not results or len(results) == 0:
            return frame, []
        result = results[0]
        if not hasattr(result, "keypoints") or result.keypoints is None:
            return frame, []
        keypoints = result.keypoints.xy.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        updated_tracks = []
        for i, (box, kps) in enumerate(zip(boxes, keypoints)):
            x1, y1, x2, y2 = box.astype(int)
            helmet_detected = self._detect_helmet_by_keypoints(frame, kps) if self.helmet_var.get() else None
            vest_detected = self._detect_vest_by_keypoints(frame, kps) if self.vest_var.get() else None

            violation_msgs = []
            is_compliant = True
            if self.helmet_var.get():
                if not helmet_detected:
                    violation_msgs.append("Thiếu mũ bảo hộ")
                    is_compliant = False
            if self.vest_var.get():
                if not vest_detected:
                    violation_msgs.append("Thiếu áo phản quang")
                    is_compliant = False

            matched = None
            max_iou = 0
            for track in self.tracked_persons:
                iou_score = iou(box, track.box)
                if iou_score > 0.5 and iou_score > max_iou:
                    max_iou = iou_score
                    matched = track
            if matched:
                changed = matched.update(box, helmet_detected, vest_detected)
            else:
                matched = PersonTrack(box, helmet_detected, vest_detected)
                changed = True
            if not matched.counted or changed:
                matched.counted = True
                if is_compliant:
                    self.stats['total_compliant'] += 1
                else:
                    self.stats['total_violations'] += 1
                self.stats['total_detections'] += 1
            updated_tracks.append(matched)

            color = (0, 255, 0) if is_compliant else (0, 0, 255)
            if is_compliant:
                label = "TUÂN THỦ PPE"
            else:
                label = " - ".join(violation_msgs)
            if self.boxes_var.get():
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            self._draw_pose(frame, kps)
            if self.helmet_var.get():
                self._draw_head_region(frame, kps, color)
            if self.vest_var.get():
                self._draw_torso_region(frame, kps, color)
            if self.helmet_var.get() and helmet_detected:
                self._draw_helmet_line(frame, kps, color)
            if self.vest_var.get() and vest_detected:
                self._draw_vest_line(frame, kps, color)
            if not is_compliant and self.sound_var.get():
                self._play_alert()
        self.tracked_persons = updated_tracks
        return frame, updated_tracks

    def _draw_head_region(self, frame, kps, color):
        head_pts = kps[[0, 1, 2, 5, 6]]
        x_min, y_min = np.min(head_pts, axis=0).astype(int)
        x_max, y_max = np.max(head_pts, axis=0).astype(int)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(frame, "Đầu", (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def _draw_torso_region(self, frame, kps, color):
        torso_pts = kps[[5, 6, 11, 12]]
        x_min, y_min = np.min(torso_pts, axis=0).astype(int)
        x_max, y_max = np.max(torso_pts, axis=0).astype(int)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(frame, "Thân", (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def _draw_helmet_line(self, frame, kps, color):
        pt1 = tuple(map(int, kps[1]))
        pt2 = tuple(map(int, kps[5]))
        pt_head = tuple(map(int, kps[0]))
        cv2.line(frame, pt1, pt_head, color, 4)
        cv2.line(frame, pt2, pt_head, color, 4)
        cv2.putText(frame, "Mũ bảo hộ", pt_head, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def _draw_vest_line(self, frame, kps, color):
        pt1 = tuple(map(int, kps[5]))
        pt2 = tuple(map(int, kps[6]))
        pt3 = tuple(map(int, kps[11]))
        pt4 = tuple(map(int, kps[12]))
        cv2.line(frame, pt1, pt3, color, 4)
        cv2.line(frame, pt2, pt4, color, 4)
        cv2.putText(frame, "Áo phản quang", pt3, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def _draw_pose(self, frame, kps):
        skeleton = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 11), (11, 12), (12, 13), (13, 14),
            (6, 12), (12, 13), (13, 14), (14, 15),
            (5, 6), (11, 12)
        ]
        for x, y in kps:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
        for i, j in skeleton:
            if i < len(kps) and j < len(kps):
                pt1 = tuple(map(int, kps[i]))
                pt2 = tuple(map(int, kps[j]))
                cv2.line(frame, pt1, pt2, (255, 0, 255), 2)

    def _update_video_display(self, frame):
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            display_size = (640, 480)
            pil_image = pil_image.resize(display_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(pil_image)
            self.video_label.config(image=photo, text="")
            self.video_label.image = photo
        except Exception as e:
            self._log_message(f"Display error: {e}")

    def _detect_image(self):
        try:
            image = cv2.imread(self.video_source)
            if image is None:
                messagebox.showerror("Lỗi", "Không thể đọc file ảnh!")
                return
            image = cv2.resize(image, (640, 480))
            annotated_image, persons = self._detect_and_draw(image)
            self._update_video_display(annotated_image)
            self._update_session_stats()
            self._log_message("🖼️ Đã phân tích ảnh")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi xử lý ảnh: {e}")

    def _update_session_stats(self):
        if self.stats['session_start']:
            elapsed = datetime.now() - self.stats['session_start']
            self.session_time_var.set(str(elapsed).split('.')[0])
        self.detections_var.set(str(self.stats['total_detections']))
        self.violations_var.set(str(self.stats['total_violations']))
        self.compliant_var.set(str(self.stats['total_compliant']))
        total = self.stats['total_detections']
        if total > 0:
            compliance_rate = (self.stats['total_compliant'] / total) * 100
            self.compliance_rate_var.set(f"{compliance_rate:.1f}%")
        else:
            self.compliance_rate_var.set("0%")

    def _play_alert(self):
        try:
            if self.alert_sound:
                self.alert_sound.play()
        except Exception as e:
            self._log_message(f"Sound error: {e}")

    def _log_message(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        if int(self.log_text.index('end-1c').split('.')[0]) > 1000:
            self.log_text.delete('1.0', '500.0')

    def _update_status(self, status_text, color):
        self.status_var.set(status_text)
        self.status_label.config(foreground=color)

    def _update_button_states(self, start=None, pause=None, stop=None):
        if start is not None:
            self.start_btn.config(state="normal" if start else "disabled")
        if pause is not None:
            self.pause_btn.config(state="normal" if pause else "disabled")
        if stop is not None:
            self.stop_btn.config(state="normal" if stop else "disabled")

    def _clear_log(self):
        self.log_text.delete('1.0', tk.END)
        self._log_message("🗑️ Log đã được xóa")

    def _reset_stats(self):
        self.stats = {
            'total_detections': 0,
            'total_violations': 0,
            'total_compliant': 0,
            'session_start': None
        }
        self.detections_var.set("0")
        self.violations_var.set("0")
        self.compliant_var.set("0")
        self.compliance_rate_var.set("0%")
        self.session_time_var.set("00:00:00")
        self.tracked_persons = []
        self._log_message("🔄 Đã reset thống kê")

    def _show_overview_stats(self):
        overview_window = tk.Toplevel(self.root)
        overview_window.title("📈 Thống kê tổng quan PPE Detection")
        overview_window.geometry("600x400")
        overview_window.configure(bg='#f0f0f0')
        main_frame = ttk.Frame(overview_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        ttk.Label(main_frame, text="📈 THỐNG KÊ TỔNG QUAN",
                  font=("Arial", 16, "bold")).pack(pady=(0, 20))
        stats_frame = ttk.LabelFrame(main_frame, text="📊 Số liệu phiên hiện tại", padding=15)
        stats_frame.pack(fill=tk.X, pady=10)
        stats_data = [
            ("⏰ Thời gian hoạt động:", self.session_time_var.get()),
            ("👥 Tổng số người phát hiện:", str(self.stats['total_detections'])),
            ("✅ Số trường hợp tuân thủ:", str(self.stats['total_compliant'])),
            ("⚠️ Số vi phạm:", str(self.stats['total_violations'])),
            ("📊 Tỷ lệ tuân thủ:", self.compliance_rate_var.get()),
        ]
        for i, (label, value) in enumerate(stats_data):
            ttk.Label(stats_frame, text=label, font=("Arial", 10)).grid(
                row=i, column=0, sticky="w", padx=(0, 20), pady=5)
            color = "green" if "tuân thủ" in label else "red" if "vi phạm" in label else "black"
            ttk.Label(stats_frame, text=value, font=("Arial", 10, "bold"),
                      foreground=color).grid(row=i, column=1, sticky="w", pady=5)
        settings_frame = ttk.LabelFrame(main_frame, text="⚙️ Cài đặt hiện tại", padding=15)
        settings_frame.pack(fill=tk.X, pady=10)
        settings_data = [
            ("👤 Nhận diện người:", "Bật" if self.person_var.get() else "Tắt"),
            ("🛡️ Phát hiện mũ bảo hiểm:", "Bật" if self.helmet_var.get() else "Tắt"),
            ("🦺 Phát hiện áo phản quang:", "Bật" if self.vest_var.get() else "Tắt"),
            ("🎯 Độ nhạy phát hiện:", self.sensitivity_var.get()),
        ]
        for i, (label, value) in enumerate(settings_data):
            ttk.Label(settings_frame, text=label).grid(row=i, column=0, sticky="w", padx=(0, 20), pady=3)
            color = "green" if value == "Bật" else "gray"
            ttk.Label(settings_frame, text=value, foreground=color).grid(row=i, column=1, sticky="w", pady=3)
        ttk.Button(main_frame, text="Đóng", command=overview_window.destroy).pack(pady=10)

    def _show_report(self):
        report_window = tk.Toplevel(self.root)
        report_window.title("📊 Báo cáo PPE Detection")
        report_window.geometry("800x600")
        notebook = ttk.Notebook(report_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        summary_frame = ttk.Frame(notebook)
        notebook.add(summary_frame, text="📈 Tổng quan")
        summary_text = tk.Text(summary_frame, wrap=tk.WORD, font=("Consolas", 10))
        summary_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        report_content = self._generate_report()
        summary_text.insert(tk.END, report_content)
        summary_text.config(state="disabled")

    def _generate_report(self):
        total_detections = max(self.stats['total_detections'], 1)
        compliance_rate = (self.stats['total_compliant'] / total_detections) * 100
        violation_rate = (self.stats['total_violations'] / total_detections) * 100
        report = f"""
📊 BÁO CÁO HỆ THỐNG PHÁT HIỆN ĐỒ BẢO HỘ LAO ĐỘNG
{'=' * 60}

📅 Thời gian tạo báo cáo: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
⏰ Thời gian phiên làm việc: {self.session_time_var.get()}

📈 THỐNG KÊ TỔNG QUAN:
• Tổng số người được phát hiện: {self.stats['total_detections']}
• Số trường hợp tuân thủ PPE: {self.stats['total_compliant']}
• Số trường hợp vi phạm: {self.stats['total_violations']}
• Tỷ lệ tuân thủ: {compliance_rate:.1f}%
• Tỷ lệ vi phạm: {violation_rate:.1f}%

🛡️ CÀI ĐẶT PPE DETECTION:
• Nhận diện người: {'Bật' if self.person_var.get() else 'Tắt'}
• Phát hiện mũ bảo hiểm: {'Bật' if self.helmet_var.get() else 'Tắt'}
• Phát hiện áo phản quang: {'Bật' if self.vest_var.get() else 'Tắt'}
• Độ nhạy phát hiện: {self.sensitivity_var.get()}

🎯 ĐÁNH GIÁ MỨC ĐỘ AN TOÀN:
"""
        if compliance_rate >= 90:
            report += "🟢 XUẤT SẮC - Mức độ tuân thủ rất cao\n"
        elif compliance_rate >= 80:
            report += "🟡 TỐT - Mức độ tuân thủ khá tốt\n"
        elif compliance_rate >= 70:
            report += "🟠 TRUNG BÌNH - Cần cải thiện\n"
        else:
            report += "🔴 YẾU - Cần hành động ngay lập tức\n"
        report += f"""
📊 PHÂN TÍCH CHI TIẾT:
• Hiệu quả detection: {'Cao' if self.stats['total_detections'] > 50 else 'Thấp'}
• Tần suất vi phạm: {'Cao' if violation_rate > 30 else 'Thấp' if violation_rate < 10 else 'Trung bình'}
• Độ ổn định hệ thống: {'Ổn định' if self.stats['total_detections'] > 0 else 'Chưa đủ dữ liệu'}

💡 KHUYẾN NGHỊ:
"""
        if violation_rate > 30:
            report += "🔸 Tăng cường training về quy định PPE\n"
            report += "🔸 Kiểm tra định kỳ đồ bảo hộ cá nhân\n"
            report += "🔸 Đặt biển báo nhắc nhở tại các vị trí quan trọng\n"
        elif violation_rate > 10:
            report += "🔸 Duy trì hệ thống giám sát hiện tại\n"
            report += "🔸 Tăng cường nhắc nhở định kỳ\n"
        else:
            report += "🔸 Duy trì mức độ tuân thủ hiện tại\n"
            report += "🔸 Khen thưởng đội ngũ có ý thức cao\n"
        report += f"""
🔧 CÀI ĐẶT HỆ THỐNG:
• Âm thanh cảnh báo: {'Bật' if self.sound_var.get() else 'Tắt'}
• Hiển thị khung: {'Có' if self.boxes_var.get() else 'Không'}

{'=' * 60}
Báo cáo được tạo tự động bởi PPE Detection System v2.0
Powered by YOLOv8 Pose + Enhanced Color Analysis
"""
        return report

def main():
    try:
        root = tk.Tk()
        app = PPEDetectionGUI(root)
        try:
            root.iconbitmap("icon.ico")
        except:
            pass
        root.eval('tk::PlaceWindow . center')
        root.mainloop()
    except Exception as e:
        print(f"Error starting application: {e}")
        messagebox.showerror("Lỗi khởi động", f"Không thể khởi động ứng dụng: {e}")

if __name__ == "__main__":
    main()