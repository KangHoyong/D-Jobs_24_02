# 실습 자료 04 PySider6 - 응용 프로그램 : 이미지 뷰어
"""
기능
- 이미지 폴더 열기, 리스트 뷰, 이미지 뷰, 뒤로가기, 앞으로 가기 버튼
"""

import os
import sys
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
    QTreeWidget,
    QHBoxLayout
)
from PySide6 import QtCore, QtGui

class MainWindow(QMainWindow) :
    def __init__(self):
        super().__init__()
        self.setWindowTitle("이미지 뷰어")
        self.resize(800, 600)

        # 버튼 생성 및 이벤트 등록
        self.folder_on_button = QPushButton("폴더 열기")
        self.folder_on_button.clicked.connect(self.open_folder_dialog)
        self.back_button = QPushButton("뒤로 가기")
        self.back_button.clicked.connect(self.go_back)
        self.forward_button = QPushButton("앞으로 가기")
        self.forward_button.clicked.connect(self.go_forward)

        # label 생성 및 이벤트 등록
        self.image_label = QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)

        # list 위젯 생성 및 등록
        self.image_list_widget = QListWidget()
        self.image_list_widget.currentRowChanged.connect(self.display_image)

        # 트리 위젯 생성 및 등록
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(["파일"])

        # left layout 구성 및 등록
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.folder_on_button)
        left_layout.addWidget(self.image_list_widget)

        # right layout 구성 및 등록
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.image_label)
        right_layout.addWidget(self.back_button)
        right_layout.addWidget(self.forward_button)

        # main layout 구성 및 등록
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 2)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.current_folder = ""
        self.current_images = []
        self.current_index = -1

    # 폴더에 있는 이미지 data를 가져오기 위한 함수
    def open_folder_dialog(self):
        folder_dialog = QFileDialog(self)
        folder_dialog.setFileMode(QFileDialog.Directory)
        folder_dialog.setOption(QFileDialog.ShowDirsOnly, True)
        folder_dialog.directoryEntered.connect(self.set_folder_path)
        folder_dialog.accepted.connect(self.load_images)
        folder_dialog.exec_()

    # 폴더 경로 위치 호출 함수
    def set_folder_path(self, folder_path):
        self.current_folder = folder_path

    def load_images(self):
        self.image_list_widget.clear()
        self.tree_widget.clear()

        if self.current_folder :
            self.current_images = []
            self.current_index = []

            image_extensions = (".jpg", ".png", ".jpeg", ".gif", ".bmp") # 이미지 타입 정의

            root_item = QTreeWidgetItem(self.tree_widget, [self.current_folder])
            self.tree_widget.addTopLevelItem(root_item)

            for dir_path, _, file_names in os.walk(self.current_folder) :
                dir_item = QTreeWidgetItem(root_item, [os.path.basename(dir_path)])
                root_item.addChild(dir_item)

                for file_name in file_names :
                    if file_name.lower().endswith(image_extensions) :
                        file_item = QTreeWidgetItem(dir_item, [file_name])
                        dir_item.addChild(file_item)
                        file_path = os.path.join(dir_path, file_name)
                        self.current_images.append(file_path)
                        self.image_list_widget.addItem(file_name)

            if self.current_images :
                self.image_list_widget.setCurrentRow(0)

    def display_image(self, index):
        if 0 <= index < len(self.current_images) :
            self.current_index = index
            image_path = self.current_images[self.current_index]
            pixmap = QtGui.QPixmap(image_path)
            scaled_pixmap = pixmap.scaled(self.image_label.size() * 0.5, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            self.image_label.setPixmap(scaled_pixmap)

    def go_back(self):
        if self.current_index > 0 :
            self.image_list_widget.setCurrentRow(self.current_index - 1)

    def go_forward(self):
        if self.current_index < len(self.current_images) -1 :
            self.image_list_widget.setCurrentRow(self.current_index + 1)
                       
if __name__ == "__main__" :
    app = QApplication(sys.argv)
    windows = MainWindow()
    windows.show()
    sys.exit(app.exec())