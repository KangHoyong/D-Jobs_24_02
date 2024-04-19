# 실습 자료 02 PySider6 - 응용 프로그램 : 시그널과 슬롯의 연결 응용 버전 실습
"""
1. 사용자로부터 나이, 성별, 국가 입력 받기
2. 보기버튼누르면입력정보를보여준다
3. 입력정보창에서는(나이성별국가)입력받은정보출력
- 추가 적으로 저장 버튼, 닫기 버튼 불러오기 버튼 추가
4. 저장된 정보 창에서는 리스트 박스를 이용하여 (중복 방지를 위한 ID 추가 여기서는 Time 함수를 이용) DB 처럼 auto number 없음 저장된 정보를 읽어 오기
- ID / 나이 / 성별 / 국가
"""

import csv
import sys
import time
from PySide6.QtWidgets import (
    QApplication,
    QPushButton,
    QWidget,
    QDialog,
    QListWidget,
    QLineEdit,
    QVBoxLayout,
    QLabel,
    QMessageBox
)

class MainWindows(QWidget) :
    def __init__(self):
        super().__init__()
        self.setWindowTitle("정보 입력")

        self.age_line_edit = QLineEdit()
        self.gender_line_edit = QLineEdit()
        self.country_line_edit = QLineEdit()
        self.view_button = QPushButton("보기")

        layout = QVBoxLayout()
        layout.addWidget(QLabel("나이 : "))
        layout.addWidget(self.age_line_edit)
        layout.addWidget(QLabel("성별 : "))
        layout.addWidget(self.gender_line_edit)
        layout.addWidget(QLabel("국가 : "))
        layout.addWidget(self.country_line_edit)
        layout.addWidget(self.view_button)

        self.setLayout(layout)

        self.view_button.clicked.connect(self.show_info)

    def show_info(self):
        age = self.age_line_edit.text()
        gender = self.gender_line_edit.text()
        country = self.country_line_edit.text()

        # 시그널과 슬롯 연결
        info_window = InfoSubWindow(age, gender, country)
        info_window.setModal(True)
        info_window.exec()

class InfoSubWindow(QDialog) :
    def __init__(self, age, gender, country):
        super().__init__()
        self.setWindowTitle("정보 확인 창")

        sub_window_layout = QVBoxLayout()
        sub_window_layout.addWidget(QLabel(f"나이 : {age}"))
        sub_window_layout.addWidget(QLabel(f"성별 : {gender}"))
        sub_window_layout.addWidget(QLabel(f"국가 : {country}"))

        save_button = QPushButton("저장")
        close_button = QPushButton("닫기")
        load_button = QPushButton("불러오기")

        sub_window_layout.addWidget(save_button)
        sub_window_layout.addWidget(close_button)
        sub_window_layout.addWidget(load_button)

        self.setLayout(sub_window_layout)

        save_button.clicked.connect(lambda : self.save_info(age, gender, country))
        load_button.clicked.connect(self.load_info)
        close_button.clicked.connect(self.close)

    # info.csv 에 데이터 저장
    def save_info(self, age, gender, country):
        data = [generate_id(), age, gender, country]
        try :
            with open('info.csv', 'a', newline="") as csvfile :
                writer = csv.writer(csvfile)
                writer.writerow(data)
            QMessageBox.information(self, "저장 완료", "정보가 성공적으로 저장 완료")

        except Exception as e :
            QMessageBox.critical(self, "저장 실패", f"정보 저장 중 오류 발생 : \n{str(e)}")

    # info.csv 에 데이터 불러오기
    def load_info(self):
        try :
            with open("info.csv", 'r') as csvfile :
                reader = csv.reader(csvfile)
                lines = [line for line in reader]

            if len(lines) > 0 :
                list_window = ListWindow(lines)
                list_window.exec()
            else :
                QMessageBox.information(self, "정보 불러오기" , "저장된 정보가 없습니다.")
        except Exception as e :
            QMessageBox.critical(self, "정보 불러오기 실패", f"정보 불러오기 중 오류 발생 : \n{str(e)}")

class ListWindow(QDialog) :
    def __init__(self, lines):
        super().__init__()
        self.setWindowTitle("저장된 정보")

        list_widget = QListWidget()
        for line in lines :
            item = f"ID {line[0]}, 나이 : {line[1]}, 성별 : {line[2]}, 국가 : {line[3]}"
            list_widget.addItem(item)

        layout = QVBoxLayout()
        layout.addWidget(list_widget)

        self.setLayout(layout)
def generate_id() :
    return str(int(time.time()))

if __name__ == "__main__"  :
    app = QApplication(sys.argv)
    windows = MainWindows()
    windows.show()
    sys.exit(app.exec())