# 실습 자료 05 PySider6 - 영상 view 프로그램 만들기
"""
기능
 - 영상 파일 읽기, 비디오 실행, 정지, 캡처
"""

import os
import sys
import cv2

from PySide6.QtWidgets import (
    QApplication,
    QPushButton,
    QWidget,
    QTreeWidgetItem,
    QFileDialog,
    QVBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QStatusBar,
    QSizePolicy
)

from PySide6.QtCore import Qt, QTimer
from PySide6 import QtGui

class MainWindow(QMainWindow) :
    def __init__(self):
        super().__init__()
        self.setWindowTitle("비디오 플레이어 및 캡처")
        self.resize(800, 600)

        # 버튼 생성 및 등록
        self.video_file_button = QPushButton("영상 파일 읽기")
        self.video_file_button.clicked.connect(self.open_video_file_dialog)

        self.play_button = QPushButton("비디오 실행")
        self.play_button.clicked.connect(self.play_video)

        self.pause_button = QPushButton("정지")
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self.pause_video)

        self.capture_button = QPushButton("캡처")
        self.capture_button.setEnabled(False)
        self.capture_button.clicked.connect(self.capture_frame)

        # 레이블 생성 및 등록
        self.video_view_label = QLabel()
        self.video_view_label.setAlignment(Qt.AlignCenter)
        self.video_view_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # main layout 생성
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_file_button)
        main_layout.addWidget(self.play_button)
        main_layout.addWidget(self.pause_button)
        main_layout.addWidget(self.capture_button)
        main_layout.addWidget(self.video_view_label)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.video_path = ""
        self.video_width = 720
        self.video_height = 640

        self.video_capture = True
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.display_next_frame)

        self.paused = False
        self.current_frame = 0
        self.capture_count = 0

        # 상태 프로세스바 생성
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def open_video_file_dialog(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("video Files (*.mp4 *.avi *.mov *.mkv)")
        if file_dialog.exec() :
            selected_files = file_dialog.selectedFiles()
            if selected_files :
                self.video_path = selected_files[0]
                self.status_bar.showMessage(f"선택된 영상 : {self.video_path}")

    def display_next_frame(self):
        if self.video_path :
            ret, frame = self.video_capture.read()

            if ret :
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = self.resize_frame(frame_rgb)
                h, w , _ = frame_resized.shape
                if w > 0 and h > 0 :
                    frame_image = QtGui.QImage(frame_resized, w, h , QtGui.QImage.Format_RGB888)
                    pixmap = QtGui.QPixmap.fromImage(frame_image)
                    self.video_view_label.setPixmap(pixmap)
                    self.video_view_label.setScaledContents(True)

                self.current_frame += 1

            else :
                self.video_timer.stop()

    def play_video(self):
        if self.video_path :
            if self.paused :
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                self.paused = True
            else :
                self.video_capture = cv2.VideoCapture(self.video_path)
                self.current_frame = 0

            self.play_button.setEnabled(False)
            self.pause_button.setEnabled(True)
            self.capture_button.setEnabled(True)
            self.video_timer.start(30)

    def pause_video(self):
        self.video_timer.stop()
        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.capture_button.setEnabled(not self.paused) # 활성화 여부 update
        self.paused = True

    def capture_frame(self):
        if not self.paused :
            return

        ret, frame = self.video_capture.read()

        if ret :
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = self.resize_frame(frame_rgb)
            h, w, _ = frame_resized.shape
            if w > 0 and h > 0 :
                folder_name = os.path.splitext(os.path.basename(self.video_path))[0]
                file_name = f"{folder_name}_{self.capture_count:04d}_image.png"
                file_path = os.path.join("./data/", file_name)
                os.makedirs("./data", exist_ok=True)
                cv2.imwrite(file_path, frame_resized)
                self.capture_count += 1
                self.status_bar.showMessage(f"캡처 완료 : {file_path}")

    def closeEvent(self, event):
        self.video_timer.stop()
        if self.video_capture :
            self.video_capture.release()

        event.accept()

    def resize_frame(self, frame):
        height, width, _ = frame.shape

        # 새로운 프레임을 저장할 변수 생성
        resize_frame = frame.copy()

        if width > self.video_width :
            ratio = self.video_width / width
            frame = cv2.resize(resize_frame, (self.video_width, int(height * ratio)))

        if height > self.video_height :
            ratio = self.video_height / height
            frame = cv2.resize(resize_frame, (int(width * ratio), self.video_height))

        return frame

if __name__ == "__main__" :
    app = QApplication(sys.argv)
    windows = MainWindow()
    windows.show()
    sys.exit(app.exec())
