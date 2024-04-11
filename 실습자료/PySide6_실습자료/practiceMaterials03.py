# 실습 자료 03 PySider6 - 응용 프로그램 : 그룹 상자를 이용한 복잡한 UI 실습

import csv
import sys
from PySide6.QtWidgets import (
    QApplication,
    QPushButton,
    QWidget,
    QGroupBox,
    QLineEdit,
    QVBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox
)

class MainWindow(QWidget) :
    def __init__(self):
        super().__init__()
        self.setWindowTitle("복잡한 UI 응용")
        self.resize(500, 200)

        # Group box 생성
        group_box1 = QGroupBox("info")
        group_box2 = QGroupBox("입력 내용 보기")
        group_box3 = QGroupBox("저장 및 불러오기")

        # label 생성
        self.label_id = QLabel("id : ")
        self.label_age = QLabel("나이 : ")
        self.label_gender = QLabel("성별 : ")
        self.label_country = QLabel("국가 : ")

        # line edit 생성
        self.id_line_edit = QLineEdit()
        self.age_line_edit = QLineEdit()
        self.gender_line_edit = QLineEdit()
        self.country_line_edit = QLineEdit()

        # 버튼 생성 및 등록 작업
        self.view_button = QPushButton("보기")
        self.view_button.clicked.connect(self.show_info)
        self.clear_button = QPushButton("초기화")
        self.clear_button.clicked.connect(self.clear_info)
        self.save_button = QPushButton("저장")
        self.save_button.clicked.connect(self.save_info)
        self.load_button = QPushButton("불러오기")
        self.load_button.clicked.connect(self.load_info)

        # 레이아웃 관리자 소개 (박스 레이아웃, 그리드 레이아웃) 추가 실습
        self.list_widget = QListWidget() # 리스트 웨젯 등록

        # 1번 layout 설정
        layout1 = QVBoxLayout()
        layout1.addWidget(self.label_id)
        layout1.addWidget(self.id_line_edit)
        layout1.addWidget(self.label_age)
        layout1.addWidget(self.age_line_edit)
        layout1.addWidget(self.label_gender)
        layout1.addWidget(self.gender_line_edit)
        layout1.addWidget(self.label_country)
        layout1.addWidget(self.country_line_edit)

        # Group box layout1번에 등록
        group_box1.setLayout(layout1)

        # 2번 layout 설정
        self.info_label = QLabel()
        layout2 = QVBoxLayout()
        layout2.addWidget(self.info_label)
        layout2.addWidget(self.view_button)
        layout2.addWidget(self.clear_button)
        layout2.setContentsMargins(10, 10, 10, 10)

        # Group box layout2번에 등록
        group_box2.setLayout(layout2)

        # 3번 layout 설정
        layout3 = QVBoxLayout()
        layout3.addWidget(self.save_button)
        layout3.addWidget(self.load_button)
        layout3.addWidget(self.list_widget)
        layout3.setContentsMargins(10, 10, 10, 10)
        group_box3.setLayout(layout3)

        # main layout에 등록
        main_layout = QVBoxLayout()
        main_layout.addWidget(group_box1)
        main_layout.addWidget(group_box2)
        main_layout.addWidget(group_box3)

        self.setLayout(main_layout)

    # show_info : 입려된 id, age, gender, country 정보 출력
    def show_info(self):
        id = self.id_line_edit.text()
        age = self.age_line_edit.text()
        gender = self.gender_line_edit.text()
        country = self.country_line_edit.text()

        info_text = f"아이디 : {id} \n나이 : {age} \n성별 : {gender} \n국가 : {country}"
        self.info_label.setText(info_text)

    # clear_info : 입력된 id, gender, country 정보 초기화
    def clear_info(self):
        self.id_line_edit.clear()
        self.age_line_edit.clear()
        self.gender_line_edit.clear()
        self.country_line_edit.clear()
        self.info_label.clear()

    # save_info : id, gender, country 정보를 저장
    def save_info(self):
        id = self.id_line_edit.text()
        age = self.age_line_edit.text()
        gender = self.gender_line_edit.text()
        country = self.country_line_edit.text()

        data = [id, age, gender, country]

        try :
            with open('data.csv', 'a', newline='', encoding='utf-8') as file :
                writer = csv.writer(file)
                writer.writerow(data)

            QMessageBox.information(self, "저장 완료", "입력 내용이 성공적으로 저장되었습니다.")

        except Exception as e :
            QMessageBox.critical(self, "저장 실패", f"입력 내용 중 오류가 발생 : \n{str(e)}")

    # load_info : id, gender, country 저장된 정보 가져오기
    def load_info(self):
        self.list_widget.clear()

        try :
            with (open("data.csv", 'r') as file) :
                reader = csv.reader(file)

                for row in reader :
                    data_text = f"id : {row[0]}, 나이 : {row[1]}, 성별 {row[2]}, 국가 {row[3]}"
                    self.list_widget.addItem(data_text)

        except Exception as e :
            QMessageBox.critical(self, "불러오기 실패", f"데이터 불러오기 중 오류 발생 : \n {str(e)}")


if __name__ == "__main__" :
    app = QApplication(sys.argv)
    windows = MainWindow()
    windows.show()
    sys.exit(app.exec())