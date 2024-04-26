import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
from PySide6.QtCore import Slot, QTimer
import cv2
from PIL import ImageQt, Image
from PySide6.QtGui import QPixmap

class WebcamApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.capture = None

        self.setWindowTitle("Webcam Capture")
        self.setFixedSize(1280, 720)

        self.start_button = QPushButton("Start Webcam")
        self.start_button.clicked.connect(self.start_webcam)

        self.capture_button = QPushButton("Capture Image")
        self.capture_button.clicked.connect(self.capture_image)
        self.capture_button.setEnabled(False)

        self.image_label = QLabel()

        layout = QVBoxLayout()
        layout.addWidget(self.start_button)
        layout.addWidget(self.capture_button)
        layout.addWidget(self.image_label)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.timer = QTimer()

    @Slot()
    def start_webcam(self):
        self.capture = cv2.VideoCapture(0)
        self.timer.timeout.connect(self.display_webcam)
        self.timer.start(30)  # 30ms 마다 웹캠 업데이트
        self.start_button.setEnabled(False)
        self.capture_button.setEnabled(True)

    @Slot()
    def display_webcam(self):
        ret, frame = self.capture.read()
        if ret:
            # numpy -> PIL -> Qt 이미지 변환
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)  # NumPy 배열을 PIL 이미지로 변환
            image = ImageQt.ImageQt(image)  # PIL 이미지를 Qt 이미지로 변환
            pixmap = QPixmap.fromImage(image)
            self.image_label.setPixmap(pixmap)

    @Slot()
    def capture_image(self):
        ret, frame = self.capture.read()
        if ret:
            cv2.imwrite("captured_image.jpg", frame)

    def closeEvent(self, event):
        if self.capture is not None:
            self.capture.release()
        self.timer.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WebcamApp()
    window.show()
    sys.exit(app.exec())
