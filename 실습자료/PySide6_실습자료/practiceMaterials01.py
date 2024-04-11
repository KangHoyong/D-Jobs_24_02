
import sys
import os

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QFileDialog,
    QTreeWidget,
    QVBoxLayout,
    QWidget,
    QTreeWidgetItem
)

class MainWindow(QMainWindow) :
    def __init__(self):
        super().__init__()

        self.setWindowTitle("탐색기")
        self.resize(500, 400)

        self.folder_button = QPushButton("폴더 열기")
        self.folder_button.clicked.connect(self.open_folder_dialog)

        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(["파일"])

        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.folder_button)
        self.main_layout.addWidget(self.tree_widget)

        central_widget = QWidget()
        central_widget.setLayout(self.main_layout)
        self.setCentralWidget(central_widget)

        self.folder_path = ""

    # 폴더 다이얼로그 이벤트 함수
    def open_folder_dialog(self):
        folder_dialog = QFileDialog(self)
        folder_dialog.setFileMode(QFileDialog.Directory)
        folder_dialog.setOption(QFileDialog.ShowDirsOnly, True)
        folder_dialog.directoryEntered.connect(self.set_folder_path)
        folder_dialog.accepted.connect(self.display_files)
        folder_dialog.exec_()

    # 현재 폴더 경로 가져오기
    def set_folder_path(self, folder_path):
        self.folder_path = folder_path

    def display_files(self):
        if self.folder_path :
            self.tree_widget.clear()

            root_item = QTreeWidgetItem(self.tree_widget, [self.folder_path])
            self.tree_widget.addTopLevelItem(root_item)

            for dir_path, _, file_names in os.walk(self.folder_path) :
                dir_item = QTreeWidgetItem(root_item, [os.path.basename(dir_path)])
                root_item.addChild(dir_item)

                for file_name in file_names :
                    file_item = QTreeWidgetItem(dir_item, [file_name])
                    dir_item.addChild(file_item)

                root_item.setExpanded(True)

if __name__ == "__main__" :
    app = QApplication(sys.argv)
    windows = MainWindow()
    windows.show()
    sys.exit(app.exec())