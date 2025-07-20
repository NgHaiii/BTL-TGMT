import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import time
import csv
import os
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
import pygame
from datetime import datetime
import pandas as pd


class PPEDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🛡️ PPE Detection System - Hệ thống phát hiện đồ bảo hộ lao động")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')

        # Core variables
        self.video_source = None
        self.cap = None
        self.model = None
        self.is_running = False
        self.is_paused = False
        self.detection_thread = None

        # Statistics
        self.stats = {
            'total_detections': 0,
            'total_violations': 0,
            'total_compliant': 0,
            'session_start': None
        }

        # Enhanced detection settings for better accuracy
        self.detection_config = {
            'confidence_threshold': 0.7,  # Tăng từ 0.65
            'min_detection_area': 1500,  # Tăng để loại bỏ detection nhỏ
            'enable_sound': True,
            'save_violations': True,
            'show_boxes': True,
            'helmet_detection': True,
            'vest_detection': True,
            'distance_monitoring': False,
            'strict_mode': False
        }

        # Color detection thresholds - được tối ưu
        self.color_thresholds = {
            'helmet': {
                'min_area_ratio': 0.15,  # Tăng từ 0.1
                'saturation_min': 120,  # Tăng để chỉ nhận màu sáng
                'value_min': 150,  # Tăng brightness threshold
                'contour_min_area': 500  # Loại bỏ contour nhỏ
            },
            'vest': {
                'min_area_ratio': 0.25,  # Tăng từ 0.2
                'saturation_min': 150,  # Chỉ nhận màu neon
                'value_min': 180,  # Rất sáng
                'contour_min_area': 800  # Contour lớn hơn
            }
        }

        # Results directory
        self.results_dir = "ppe_detection_results"
        os.makedirs(self.results_dir, exist_ok=True)

        # Initialize components
        self._init_sound()
        self._init_gui()
        self._load_model()

    def _init_sound(self):
        """Initialize sound system"""
        try:
            pygame.mixer.init()
            # Create alert sound
            sample_rate, duration, frequency = 44100, 0.5, 800
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = np.sin(2 * np.pi * frequency * t) * 0.3
            audio = (audio * 32767).astype(np.int16)
            stereo_audio = np.array([audio, audio]).T
            self.alert_sound = pygame.sndarray.make_sound(stereo_audio)
        except Exception as e:
            print(f"Sound initialization error: {e}")
            self.alert_sound = None

    def _init_gui(self):
        """Initialize GUI components"""
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Title
        self._create_title(main_container)

        # Content area
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Panels
        self._create_control_panel(content_frame)
        self._create_video_panel(content_frame)
        self._create_stats_panel(content_frame)

    def _create_title(self, parent):
        """Create title section"""
        title_frame = ttk.Frame(parent)
        title_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(title_frame, text="🛡️ HỆ THỐNG AN TOÀN LAO ĐỘNG",
                  font=("Arial", 18, "bold")).pack()
    def _create_control_panel(self, parent):
        """Create control panel with scrolling"""
        left_panel = ttk.LabelFrame(parent, text="🎛️ Điều khiển", padding=5, width=280)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)

        # Create scrollable frame
        canvas = tk.Canvas(left_panel)
        scrollbar = ttk.Scrollbar(left_panel, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind("<Configure>",
                              lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Add control sections
        self._add_source_selection(scrollable_frame)
        self._add_control_buttons(scrollable_frame)
        self._add_ppe_options(scrollable_frame)
        self._add_sensitivity_settings(scrollable_frame)
        self._add_general_settings(scrollable_frame)
        self._add_action_buttons(scrollable_frame)

        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Mouse wheel binding
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

    def _add_source_selection(self, parent):
        """Add source selection section"""
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
        """Add control buttons section"""
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
        """Add PPE detection options"""
        ppe_frame = ttk.LabelFrame(parent, text="🛡️ PPE Detection", padding=5)
        ppe_frame.pack(fill=tk.X, pady=5)

        self.helmet_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ppe_frame, text="⛑️ Phát hiện mũ bảo hiểm",
                        variable=self.helmet_var,
                        command=self._update_detection_settings).pack(anchor=tk.W)

        self.vest_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ppe_frame, text="🦺 Phát hiện áo phản quang",
                        variable=self.vest_var,
                        command=self._update_detection_settings).pack(anchor=tk.W)

        self.distance_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ppe_frame, text="📏 Giám sát khoảng cách",
                        variable=self.distance_var,
                        command=self._update_detection_settings).pack(anchor=tk.W)

        self.strict_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ppe_frame, text="🔒 Chế độ nghiêm ngặt",
                        variable=self.strict_var,
                        command=self._update_detection_settings).pack(anchor=tk.W)

    def _add_sensitivity_settings(self, parent):
        """Add sensitivity settings"""
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
        """Add general settings"""
        general_frame = ttk.LabelFrame(parent, text="🔧 Cài đặt chung", padding=5)
        general_frame.pack(fill=tk.X, pady=5)

        self.sound_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(general_frame, text="🔊 Âm thanh",
                        variable=self.sound_var).pack(anchor=tk.W)

        self.save_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(general_frame, text="💾 Lưu kết quả",
                        variable=self.save_var).pack(anchor=tk.W)

        self.boxes_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(general_frame, text="📦 Hiện khung",
                        variable=self.boxes_var).pack(anchor=tk.W)

    def _add_action_buttons(self, parent):
        """Add action buttons"""
        action_frame = ttk.LabelFrame(parent, text="📋 Hành động", padding=5)
        action_frame.pack(fill=tk.X, pady=5)

        actions = [
            ("📈 Thống kê tổng quan", self._show_overview_stats),
            ("📊 Báo cáo chi tiết", self._show_report),
            ("📁 Mở thư mục", self._open_results_folder),
            ("🔄 Reset", self._reset_stats)
        ]

        for text, command in actions:
            ttk.Button(action_frame, text=text, command=command).pack(fill=tk.X, pady=1)

    def _create_video_panel(self, parent):
        """Create video display panel"""
        center_panel = ttk.LabelFrame(parent, text="📺 Video Display", padding=10)
        center_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.video_frame = ttk.Frame(center_panel)
        self.video_frame.pack(fill=tk.BOTH, expand=True)

        self.video_label = ttk.Label(self.video_frame,
                                     text="📷 Video sẽ hiển thị ở đây\nNhấn 'Bắt đầu' để khởi động",
                                     font=("Arial", 12), background="black",
                                     foreground="white", anchor="center")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Status bar
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
        """Create statistics panel"""
        right_panel = ttk.LabelFrame(parent, text="📊 Thống kê", padding=10, width=350)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        right_panel.pack_propagate(False)

        # Session stats
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

        # Live log
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
        """Load YOLO model"""
        try:
            self.model = YOLO('yolov8n.pt')
            self._log_message("✅ Model YOLO đã tải thành công")
            self._log_message("ℹ️ Sử dụng YOLOv8n để phát hiện person")
            self._log_message("⚠️ PPE detection dựa trên phân tích màu sắc nâng cao")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải model: {e}")
            self._log_message(f"❌ Lỗi tải model: {e}")

    def _select_file(self):
        """Select video or image file"""
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

    def _update_detection_settings(self):
        """Update detection settings"""
        self.detection_config.update({
            'helmet_detection': self.helmet_var.get(),
            'vest_detection': self.vest_var.get(),
            'distance_monitoring': self.distance_var.get(),
            'strict_mode': self.strict_var.get()
        })

        self._log_message(f"🔧 Cập nhật PPE: Helmet={self.detection_config['helmet_detection']}, "
                          f"Vest={self.detection_config['vest_detection']}")

    def _update_sensitivity(self):
        """Update detection sensitivity"""
        sensitivity = self.sensitivity_var.get()

        sensitivity_configs = {
            'high': {
                'confidence_threshold': 0.5,
                'min_detection_area': 800,
                'helmet_min_area_ratio': 0.08,
                'vest_min_area_ratio': 0.15
            },
            'normal': {
                'confidence_threshold': 0.7,
                'min_detection_area': 1500,
                'helmet_min_area_ratio': 0.15,
                'vest_min_area_ratio': 0.25
            },
            'low': {
                'confidence_threshold': 0.85,
                'min_detection_area': 2500,
                'helmet_min_area_ratio': 0.25,
                'vest_min_area_ratio': 0.35
            }
        }

        config = sensitivity_configs[sensitivity]
        self.detection_config.update(config)
        self.color_thresholds['helmet']['min_area_ratio'] = config['helmet_min_area_ratio']
        self.color_thresholds['vest']['min_area_ratio'] = config['vest_min_area_ratio']

        self._log_message(f"🎯 Đổi độ nhạy: {sensitivity} (conf={config['confidence_threshold']:.2f})")

    def _start_detection(self):
        """Start detection process"""
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

        self._update_status("▶️ Đang chạy", "green")
        self._update_button_states(start=False, pause=True, stop=True)

        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()

        self._log_message("🚀 Bắt đầu phát hiện PPE...")

    def _pause_detection(self):
        """Pause/resume detection"""
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
        """Stop detection process"""
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
        """Main detection loop"""
        self.cap = cv2.VideoCapture(self.video_source)

        if not self.cap.isOpened():
            messagebox.showerror("Lỗi", "Không thể mở nguồn video!")
            self._stop_detection()
            return

        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        last_save_time = time.time()

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

            # Enhanced detection
            detections = self._detect_ppe_enhanced(display_frame)
            annotated_frame = self._draw_detections(display_frame.copy(), detections)
            self._update_video_display(annotated_frame)

            # Update progress
            if total_frames > 0:
                self.progress_var.set(f"{frame_count}/{total_frames}")
            else:
                self.progress_var.set(f"{frame_count}")

            # Update statistics
            self._update_session_stats(detections)

            # Save results periodically
            if time.time() - last_save_time > 5:
                self._log_detection_results(detections, frame_count)
                last_save_time = time.time()

            time.sleep(1 / 30)

        self.cap.release()
        self._log_message("🔚 Kết thúc phiên phát hiện")

    def _detect_ppe_enhanced(self, frame):
        """Enhanced PPE detection with improved accuracy"""
        try:
            results = self.model(frame, conf=self.detection_config['confidence_threshold'],
                                 verbose=False)

            detections = {
                'persons': [],
                'violations': [],
                'compliant': [],
                'total_objects': 0
            }

            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()

                    for box, conf, cls in zip(boxes, confidences, classes):
                        class_id = int(cls)
                        class_name = self.model.names[class_id]

                        if class_name == 'person':
                            # Filter by area
                            box_area = (box[2] - box[0]) * (box[3] - box[1])
                            if box_area < self.detection_config['min_detection_area']:
                                continue

                            detection_info = {
                                'bbox': box,
                                'confidence': conf,
                                'class': class_name,
                                'class_id': class_id
                            }

                            detections['persons'].append(detection_info)

                            # Enhanced PPE analysis
                            ppe_analysis = self._analyze_ppe_enhanced(frame, box)

                            if ppe_analysis['compliant']:
                                detections['compliant'].append({
                                    'person': detection_info,
                                    'ppe_status': ppe_analysis
                                })
                            else:
                                violation = {
                                    'person': detection_info,
                                    'ppe_status': ppe_analysis,
                                    'violation_type': ppe_analysis['missing_items'],
                                    'severity': ppe_analysis['severity']
                                }
                                detections['violations'].append(violation)

                        detections['total_objects'] += 1

            return detections

        except Exception as e:
            self._log_message(f"❌ Lỗi detection: {e}")
            return {'persons': [], 'violations': [], 'compliant': [], 'total_objects': 0}

    def _analyze_ppe_enhanced(self, frame, bbox):
        """Enhanced PPE analysis with better accuracy"""
        x1, y1, x2, y2 = bbox.astype(int)
        person_roi = frame[y1:y2, x1:x2]

        if person_roi.size == 0:
            return self._create_ppe_result(False, False, False, ['Không phát hiện được người'], 'high')

        has_helmet = False
        has_vest = False
        missing_items = []

        # Enhanced helmet detection
        if self.detection_config['helmet_detection']:
            head_height = int((y2 - y1) * 0.35)  # Increase head region
            head_roi = person_roi[:head_height, :]
            has_helmet = self._detect_helmet_enhanced(head_roi)
            if not has_helmet:
                missing_items.append('Mũ bảo hiểm')

        # Enhanced vest detection
        if self.detection_config['vest_detection']:
            torso_start = int((y2 - y1) * 0.25)
            torso_end = int((y2 - y1) * 0.80)
            torso_roi = person_roi[torso_start:torso_end, :]
            has_vest = self._detect_vest_enhanced(torso_roi)
            if not has_vest:
                missing_items.append('Áo phản quang')

        # Compliance logic
        compliant = self._determine_compliance(has_helmet, has_vest)

        # Calculate severity
        severity = 'high' if len(missing_items) >= 2 else 'medium' if len(missing_items) == 1 else 'none'

        return self._create_ppe_result(compliant, has_helmet, has_vest, missing_items, severity)

    def _detect_helmet_enhanced(self, head_roi):
        """Enhanced helmet detection with better color analysis"""
        if head_roi.size == 0:
            return False

        hsv = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)

        # Define enhanced helmet color ranges
        helmet_colors = [
            # Yellow (more restrictive)
            ([20, 120, 150], [30, 255, 255]),
            # White (brighter)
            ([0, 0, 200], [180, 25, 255]),
            # Red
            ([0, 150, 100], [10, 255, 255]),
            ([170, 150, 100], [180, 255, 255]),
            # Blue
            ([100, 150, 100], [130, 255, 255]),
            # Orange
            ([10, 150, 150], [20, 255, 255])
        ]

        combined_mask = np.zeros(head_roi.shape[:2], dtype=np.uint8)

        for lower, upper in helmet_colors:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Enhanced morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        # Advanced contour analysis
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return False

        # Find the largest valid contour
        valid_contours = []
        total_area = head_roi.shape[0] * head_roi.shape[1]

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.color_thresholds['helmet']['contour_min_area']:
                continue

            # Check aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0

            if 0.3 <= aspect_ratio <= 4.0:  # Reasonable helmet shape
                valid_contours.append(contour)

        if not valid_contours:
            return False

        # Calculate area ratio of largest valid contour
        largest_contour = max(valid_contours, key=cv2.contourArea)
        helmet_area = cv2.contourArea(largest_contour)
        area_ratio = helmet_area / total_area

        return area_ratio >= self.color_thresholds['helmet']['min_area_ratio']

    def _detect_vest_enhanced(self, torso_roi):
        """Enhanced vest detection with better color analysis"""
        if torso_roi.size == 0:
            return False

        hsv = cv2.cvtColor(torso_roi, cv2.COLOR_BGR2HSV)

        # Enhanced vest color ranges (high-visibility only)
        vest_colors = [
            # High-vis Orange
            ([10, 180, 180], [20, 255, 255]),
            # High-vis Yellow
            ([22, 180, 180], [35, 255, 255]),
            # High-vis Lime Green
            ([45, 180, 180], [75, 255, 255]),
            # High-vis Red
            ([0, 180, 150], [8, 255, 255]),
            ([172, 180, 150], [180, 255, 255])
        ]

        combined_mask = np.zeros(torso_roi.shape[:2], dtype=np.uint8)

        for lower, upper in vest_colors:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Enhanced morphological operations for vest patterns
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        # Look for reflective strip patterns
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        horizontal_strips = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, horizontal_kernel)

        # Combine with general vest detection
        combined_mask = cv2.bitwise_or(combined_mask, horizontal_strips)

        # Advanced contour analysis
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return False

        total_area = torso_roi.shape[0] * torso_roi.shape[1]
        valid_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.color_thresholds['vest']['contour_min_area']:
                valid_area += area

        area_ratio = valid_area / total_area
        return area_ratio >= self.color_thresholds['vest']['min_area_ratio']

    def _determine_compliance(self, has_helmet, has_vest):
        """Determine PPE compliance based on settings"""
        if not self.detection_config['helmet_detection'] and not self.detection_config['vest_detection']:
            return True

        if self.detection_config['strict_mode']:
            # Strict mode: need ALL enabled PPE
            helmet_ok = has_helmet if self.detection_config['helmet_detection'] else True
            vest_ok = has_vest if self.detection_config['vest_detection'] else True
            return helmet_ok and vest_ok
        else:
            # Normal mode: need AT LEAST ONE enabled PPE
            helmet_ok = has_helmet if self.detection_config['helmet_detection'] else False
            vest_ok = has_vest if self.detection_config['vest_detection'] else False
            return helmet_ok or vest_ok

    def _create_ppe_result(self, compliant, has_helmet, has_vest, missing_items, severity):
        """Create PPE analysis result"""
        return {
            'compliant': compliant,
            'has_helmet': has_helmet,
            'has_vest': has_vest,
            'missing_items': missing_items,
            'severity': severity
        }

    def _draw_detections(self, frame, detections):
        """Draw detection results on frame"""
        if not self.detection_config['show_boxes']:
            return frame

        # Draw compliant persons (GREEN)
        for compliant in detections['compliant']:
            person = compliant['person']
            x1, y1, x2, y2 = person['bbox'].astype(int)
            confidence = person['confidence']

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"✅ PPE OK: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # PPE status indicators
            ppe_status = compliant['ppe_status']
            status_text = ""
            if ppe_status['has_helmet']:
                status_text += "⛑️ "
            if ppe_status['has_vest']:
                status_text += "🦺 "

            if status_text:
                cv2.putText(frame, status_text, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw violations (RED)
        for violation in detections['violations']:
            person = violation['person']
            x1, y1, x2, y2 = person['bbox'].astype(int)
            confidence = person['confidence']

            color = (0, 0, 255) if violation['severity'] == 'high' else (0, 100, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            label = f"⚠️ Vi phạm: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            missing_text = ", ".join(violation['violation_type'])
            if len(missing_text) > 20:
                missing_text = missing_text[:20] + "..."

            cv2.putText(frame, missing_text, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Add info overlay
        self._add_frame_info(frame, detections)
        return frame

    def _add_frame_info(self, frame, detections):
        """Add information overlay to frame"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        person_count = len(detections['persons'])
        compliant_count = len(detections['compliant'])
        violation_count = len(detections['violations'])

        info_text = f"Persons: {person_count} | OK: {compliant_count} | Violations: {violation_count}"
        cv2.putText(frame, info_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def _update_video_display(self, frame):
        """Update video display"""
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
        """Detect PPE in single image"""
        try:
            image = cv2.imread(self.video_source)
            if image is None:
                messagebox.showerror("Lỗi", "Không thể đọc file ảnh!")
                return

            image = cv2.resize(image, (640, 480))
            detections = self._detect_ppe_enhanced(image)
            annotated_image = self._draw_detections(image, detections)
            self._update_video_display(annotated_image)
            self._update_session_stats(detections)
            self._log_detection_results(detections, 1)
            self._log_message("🖼️ Đã phân tích ảnh")

        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi xử lý ảnh: {e}")

    def _update_session_stats(self, detections):
        """Update session statistics without duplication"""
        # Update session time
        if self.stats['session_start']:
            elapsed = datetime.now() - self.stats['session_start']
            self.session_time_var.set(str(elapsed).split('.')[0])

        # Update counters
        self.stats['total_detections'] += len(detections['persons'])
        self.stats['total_violations'] += len(detections['violations'])
        self.stats['total_compliant'] += len(detections['compliant'])

        # Update display
        self.detections_var.set(str(self.stats['total_detections']))
        self.violations_var.set(str(self.stats['total_violations']))
        self.compliant_var.set(str(self.stats['total_compliant']))

        # Calculate compliance rate
        if self.stats['total_detections'] > 0:
            compliance_rate = (self.stats['total_compliant'] / self.stats['total_detections']) * 100
            self.compliance_rate_var.set(f"{compliance_rate:.1f}%")
        else:
            self.compliance_rate_var.set("0%")

        # Play alert if violations detected
        if detections['violations'] and self.detection_config['enable_sound']:
            self._play_alert()

    def _play_alert(self):
        """Play alert sound"""
        try:
            if self.alert_sound and self.sound_var.get():
                self.alert_sound.play()
        except Exception as e:
            self._log_message(f"Sound error: {e}")

    def _log_message(self, message):
        """Log message to text widget"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"

        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)

        # Limit log size
        if int(self.log_text.index('end-1c').split('.')[0]) > 1000:
            self.log_text.delete('1.0', '500.0')

    def _log_detection_results(self, detections, frame_num):
        """Log detection results"""
        person_count = len(detections['persons'])
        violation_count = len(detections['violations'])
        compliant_count = len(detections['compliant'])

        if person_count > 0:
            self._log_message(f"Frame {frame_num}: {person_count} người | ✅{compliant_count} | ⚠️{violation_count}")

            if self.detection_config['save_violations']:
                self._save_to_csv(detections, frame_num)

    def _save_to_csv(self, detections, frame_num):
        """Save results to CSV file"""
        try:
            csv_file = os.path.join(self.results_dir, "ppe_detection_log.csv")

            # Create header if file doesn't exist
            if not os.path.exists(csv_file):
                with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Timestamp', 'Frame', 'Total_Persons', 'Compliant',
                                     'Violations', 'Compliance_Rate', 'Details'])

            # Append data
            with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                person_count = len(detections['persons'])
                violation_count = len(detections['violations'])
                compliant_count = len(detections['compliant'])

                compliance_rate = (compliant_count / person_count * 100) if person_count > 0 else 0
                details = f"{person_count} persons detected"

                if violation_count > 0:
                    details += f", {violation_count} violations"
                if compliant_count > 0:
                    details += f", {compliant_count} compliant"

                writer.writerow([timestamp, frame_num, person_count, compliant_count,
                                 violation_count, f"{compliance_rate:.1f}%", details])

        except Exception as e:
            self._log_message(f"CSV save error: {e}")

    def _update_status(self, status_text, color):
        """Update status display"""
        self.status_var.set(status_text)
        self.status_label.config(foreground=color)

    def _update_button_states(self, start=None, pause=None, stop=None):
        """Update button states"""
        if start is not None:
            self.start_btn.config(state="normal" if start else "disabled")
        if pause is not None:
            self.pause_btn.config(state="normal" if pause else "disabled")
        if stop is not None:
            self.stop_btn.config(state="normal" if stop else "disabled")

    def _clear_log(self):
        """Clear log text"""
        self.log_text.delete('1.0', tk.END)
        self._log_message("🗑️ Log đã được xóa")

    def _reset_stats(self):
        """Reset statistics"""
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

        self._log_message("🔄 Đã reset thống kê")

    def _open_results_folder(self):
        """Open results folder"""
        try:
            os.startfile(self.results_dir)
        except:
            self._log_message("❌ Không thể mở thư mục kết quả")

    def _show_overview_stats(self):
        """Show overview statistics"""
        overview_window = tk.Toplevel(self.root)
        overview_window.title("📈 Thống kê tổng quan PPE Detection")
        overview_window.geometry("600x400")
        overview_window.configure(bg='#f0f0f0')

        main_frame = ttk.Frame(overview_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        ttk.Label(main_frame, text="📈 THỐNG KÊ TỔNG QUAN",
                  font=("Arial", 16, "bold")).pack(pady=(0, 20))

        # Session stats
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

        # Settings
        settings_frame = ttk.LabelFrame(main_frame, text="⚙️ Cài đặt hiện tại", padding=15)
        settings_frame.pack(fill=tk.X, pady=10)

        settings_data = [
            ("🛡️ Phát hiện mũ bảo hiểm:", "Bật" if self.helmet_var.get() else "Tắt"),
            ("🦺 Phát hiện áo phản quang:", "Bật" if self.vest_var.get() else "Tắt"),
            ("📏 Giám sát khoảng cách:", "Bật" if self.distance_var.get() else "Tắt"),
            ("🔒 Chế độ nghiêm ngặt:", "Bật" if self.strict_var.get() else "Tắt"),
            ("🎯 Độ nhạy phát hiện:", self.sensitivity_var.get()),
        ]

        for i, (label, value) in enumerate(settings_data):
            ttk.Label(settings_frame, text=label).grid(row=i, column=0, sticky="w", padx=(0, 20), pady=3)
            color = "green" if value == "Bật" else "gray"
            ttk.Label(settings_frame, text=value, foreground=color).grid(row=i, column=1, sticky="w", pady=3)

        ttk.Button(main_frame, text="Đóng", command=overview_window.destroy).pack(pady=10)

    def _show_report(self):
        """Show detailed report"""
        report_window = tk.Toplevel(self.root)
        report_window.title("📊 Báo cáo PPE Detection")
        report_window.geometry("800x600")

        notebook = ttk.Notebook(report_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Summary tab
        summary_frame = ttk.Frame(notebook)
        notebook.add(summary_frame, text="📈 Tổng quan")

        summary_text = tk.Text(summary_frame, wrap=tk.WORD, font=("Consolas", 10))
        summary_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        report_content = self._generate_report()
        summary_text.insert(tk.END, report_content)
        summary_text.config(state="disabled")

        # Data tab
        data_frame = ttk.Frame(notebook)
        notebook.add(data_frame, text="📊 Dữ liệu")

        csv_file = os.path.join(self.results_dir, "ppe_detection_log.csv")
        if os.path.exists(csv_file):
            self._show_data_table(data_frame, csv_file)
        else:
            ttk.Label(data_frame, text="Chưa có dữ liệu").pack(pady=20)

    def _show_data_table(self, parent, csv_file):
        """Show data table from CSV"""
        try:
            df = pd.read_csv(csv_file)

            tree_frame = ttk.Frame(parent)
            tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            tree = ttk.Treeview(tree_frame, columns=list(df.columns), show='headings')

            for col in df.columns:
                tree.heading(col, text=col)
                tree.column(col, width=120)

            # Show last 100 rows
            display_df = df.tail(100) if len(df) > 100 else df
            for _, row in display_df.iterrows():
                tree.insert('', tk.END, values=list(row))

            # Scrollbars
            v_scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
            h_scrollbar = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)

            tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
            tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        except Exception as e:
            ttk.Label(parent, text=f"Lỗi đọc dữ liệu: {e}").pack(pady=20)

    def _generate_report(self):
        """Generate detailed report"""
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
• Phát hiện mũ bảo hiểm: {'Bật' if self.helmet_var.get() else 'Tắt'}
• Phát hiện áo phản quang: {'Bật' if self.vest_var.get() else 'Tắt'}
• Giám sát khoảng cách: {'Bật' if self.distance_var.get() else 'Tắt'}
• Chế độ nghiêm ngặt: {'Bật' if self.strict_var.get() else 'Tắt'}
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
• Lưu kết quả: {'Có' if self.save_var.get() else 'Không'}
• Hiển thị khung: {'Có' if self.boxes_var.get() else 'Không'}

{'=' * 60}
Báo cáo được tạo tự động bởi PPE Detection System v2.0
Powered by YOLOv8 + Enhanced Color Analysis
"""
        return report


def main():
    """Main function to run the application"""
    try:
        root = tk.Tk()
        app = PPEDetectionGUI(root)

        # Set icon if available
        try:
            root.iconbitmap("icon.ico")
        except:
            pass

        # Center window
        root.eval('tk::PlaceWindow . center')
        root.mainloop()

    except Exception as e:
        print(f"Error starting application: {e}")
        messagebox.showerror("Lỗi khởi động", f"Không thể khởi động ứng dụng: {e}")


if __name__ == "__main__":
    main()