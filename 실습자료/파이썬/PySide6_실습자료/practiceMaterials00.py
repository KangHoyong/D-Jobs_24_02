# 실습 : 간단한 이미지 데이터 관리 프로그램
# 기능 : 파일 복사 / 파일 이동
import os
import shutil
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QPushButton, QLabel, QFileDialog, QMessageBox, QProgressBar)

class FileManager(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("이미지 데이터 관리 프로그램")
        self.layout = QVBoxLayout()

        self.source_label = QLabel("원본 이미지 데이터 폴더:")
        self.destination_label = QLabel("옮길 이미지 데이터 폴더:")
        self.source_button = QPushButton("원본 이미지 폴더 선택")
        self.destination_button = QPushButton("옮길 이미지 폴더 선택")
        self.copy_button = QPushButton("이미지 파일 복사")
        self.move_button = QPushButton("이미지 파일 이동")
        self.progress_bar = QProgressBar()

        self.source_button.clicked.connect(self.get_source_folder)
        self.destination_button.clicked.connect(self.get_destination_folder)
        self.copy_button.clicked.connect(self.copy_files)
        self.move_button.clicked.connect(self.move_files)

        self.layout.addWidget(self.source_label)
        self.layout.addWidget(self.destination_label)
        self.layout.addWidget(self.source_button)
        self.layout.addWidget(self.destination_button)
        self.layout.addWidget(self.copy_button)
        self.layout.addWidget(self.move_button)
        self.layout.addWidget(self.progress_bar)

        self.source_path = ""
        self.destination_path = ""

        self.setLayout(self.layout)

    def get_source_folder(self):
        self.source_path = QFileDialog.getExistingDirectory(self, "원본 데이터 폴더 선택", os.path.expanduser('~'))
        if self.source_path:
            self.source_label.setText(f"원본 데이터 폴더 경로 : {self.source_path}")

    def get_destination_folder(self):
        self.destination_path = QFileDialog.getExistingDirectory(self, "옮길 폴더 선택", os.path.expanduser('~'))
        if self.destination_path:
            self.destination_label.setText(f"옮길 폴더 경로 : {self.destination_path}")

    def copy_files(self):
        if self.source_path and self.destination_path:
            if not os.listdir(self.source_path):
                QMessageBox.warning(self, "경고", "원본 폴더에 이미지 파일이 없습니다.")
                return

            total_files = len([filename for filename in os.listdir(self.source_path) if filename.endswith('.jpg') or filename.endswith('.png')])
            copied_files = 0

            for filename in os.listdir(self.source_path):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    shutil.copy(os.path.join(self.source_path, filename), self.destination_path)
                    copied_files += 1
                    progress = int((copied_files / total_files) * 100)
                    self.progress_bar.setValue(progress)

            QMessageBox.information(self, "완료", "이미지 파일 복사가 완료되었습니다.")

    def move_files(self):
        if self.source_path and self.destination_path:
            if not os.listdir(self.source_path):
                QMessageBox.warning(self, "경고", "원본 폴더에 이미지 파일이 없습니다.")
                return

            total_files = len([filename for filename in os.listdir(self.source_path) if filename.endswith('.jpg') or filename.endswith('.png')])
            moved_files = 0

            for filename in os.listdir(self.source_path):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    shutil.move(os.path.join(self.source_path, filename), self.destination_path)
                    moved_files += 1
                    progress = int((moved_files / total_files) * 100)
                    self.progress_bar.setValue(progress)

            QMessageBox.information(self, "완료", "이미지 파일 이동이 완료되었습니다.")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("파일 관리 프로그램")
        self.setFixedSize(900, 220)
        self.setCentralWidget(FileManager())

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
