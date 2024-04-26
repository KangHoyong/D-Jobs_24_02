# PySide6 실습 : 실제 데이터를 가지고 간단한 테이블 뷰 그리기
"""
데이터 양식
DateTime,Temperature,Humidity,Label
2023-09-01 00:00:00,26.5,23.6,0

DHT11-log-data.csv 데이터를 불러와서 GUI 화면에 테이블 뷰로 보여주기
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
from PySide6.QtWidgets import QApplication, QMainWindow, QTableView, QVBoxLayout, QWidget
from PySide6.QtCore import QAbstractTableModel, Qt, Slot

class DataTableModel(QAbstractTableModel):
    def __init__(self, data):
        super().__init__()
        self._data = data
    def rowCount(self, parent):
        return len(self._data.index)
    def columnCount(self, parent):
        return len(self._data.columns)
    def data(self, index, role):
        if role == Qt.DisplayRole:
            return str(self._data.iloc[index.row(), index.column()])
        return None
    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])
            elif orientation == Qt.Vertical:
                return str(self._data.index[section])
        return None


class MainWindow(QMainWindow):
    def __init__(self, data):
        super().__init__()
        self.data = data

        # 테이블 모델 및 뷰 설정
        self.table_view = QTableView()
        self.model = DataTableModel(self.data)
        self.table_view.setModel(self.model)
        self.table_view.setSelectionBehavior(QTableView.SelectRows)
        self.table_view.selectionModel().selectionChanged.connect(self.plot_selected_range)

        # 그래프 설정
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel('DateTime')
        self.ax.set_ylabel('Value')

        # 테이블 뷰와 그래프를 포함하는 위젯 생성
        central_widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.table_view)
        layout.addWidget(self.figure.canvas)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.setWindowTitle("CSV 데이터 테이블 및 그래프")

    @Slot()
    def plot_selected_range(self):
        selected_indexes = self.table_view.selectionModel().selectedIndexes()
        if not selected_indexes:
            return

        selected_rows = set(index.row() for index in selected_indexes)
        selected_data = self.data.iloc[list(selected_rows)]

        self.ax.clear()
        selected_data['DateTime'] = pd.to_datetime(selected_data['DateTime'])
        selected_data = selected_data.set_index('DateTime')

        # Plot only temperature and humidity columns
        selected_data[['Temperature', 'Humidity']].plot(ax=self.ax, legend=False)

        # Set labels for each line
        self.ax.lines[0].set_label('Temperature')
        self.ax.lines[1].set_label('Humidity')

        # Add legend
        self.ax.legend(loc='upper right')
        self.figure.canvas.draw()

if __name__ == "__main__":
    # CSV 파일 읽어오기
    csv_file = "./DHT11_log_data.csv"
    try:
        data = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"파일 '{csv_file}'을(를) 찾을 수 없습니다.")
        sys.exit(1)

    # Qt 애플리케이션 생성 및 실행
    app = QApplication(sys.argv)
    window = MainWindow(data)
    window.show()
    sys.exit(app.exec())
