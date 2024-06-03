# D-Jobs_24_02
D-Jobs 아카데미 2기 객체 인식 교육

## D-Jobs 24 2기 강의자료 

### 1. [강의자료-1 링크](https://github.com/KangHoyong/D-Jobs_24_02/blob/main/%EA%B0%95%EC%9D%98%EC%9E%90%EB%A3%8C/%EA%B0%9D%EC%B2%B4%20%EC%9D%B8%EC%8B%9D/%EA%B0%9D%EC%B2%B4%EC%9D%B8%EC%8B%9D%20%EA%B5%90%EC%9C%A1%EC%9E%90%EB%A3%8C%2001.pdf)

    강의 내용 
     - 객체 인식 소개 
     - 이미지 처리 픽셀 기본 정의 
     - 특징 추출 소개 

* [실습 필요한 샘플 이미지](https://github.com/KangHoyong/D-Jobs_24_02/tree/main/%EC%8B%A4%EC%8A%B5%EC%9E%90%EB%A3%8C/%EA%B0%9D%EC%B2%B4%EC%9D%B8%EC%8B%9D/%EA%B0%9D%EC%B2%B4%20%EC%9D%B8%EC%8B%9D%20%EA%B5%90%EC%9C%A1%20%EC%9E%90%EB%A3%8C%2001)

### 2. [강의자료-2 링크](https://github.com/KangHoyong/D-Jobs_24_02/blob/main/%EA%B0%95%EC%9D%98%EC%9E%90%EB%A3%8C/%EA%B0%9D%EC%B2%B4%20%EC%9D%B8%EC%8B%9D/%EA%B0%9D%EC%B2%B4%EC%9D%B8%EC%8B%9D%20%EA%B5%90%EC%9C%A1%EC%9E%90%EB%A3%8C%2002.pdf)

    강의 내용 
     - 특징 디스크립터 소개 
     - 객체 인식 파트 II

* [실습 필요한 샘플 이미지](https://github.com/KangHoyong/D-Jobs_24_02/tree/main/%EC%8B%A4%EC%8A%B5%EC%9E%90%EB%A3%8C/%EA%B0%9D%EC%B2%B4%EC%9D%B8%EC%8B%9D/%EA%B0%9D%EC%B2%B4%20%EC%9D%B8%EC%8B%9D%20%EA%B5%90%EC%9C%A1%20%EC%9E%90%EB%A3%8C%2002)

### 3. [강의자료-3 링크](https://github.com/KangHoyong/D-Jobs_24_02/blob/main/%EA%B0%95%EC%9D%98%EC%9E%90%EB%A3%8C/%EA%B0%9D%EC%B2%B4%20%EC%9D%B8%EC%8B%9D/%EA%B0%9D%EC%B2%B4%EC%9D%B8%EC%8B%9D%20%EA%B5%90%EC%9C%A1%EC%9E%90%EB%A3%8C%2003.pdf)

    강의 내용 
     - One Stage YOLO 알고리즘 소개 
     - SDD 알고리즘 소개 

### 4. [YOLOv8 학습 방법 강의 자료 링크](https://github.com/KangHoyong/D-Jobs_24_02/blob/main/%EA%B0%95%EC%9D%98%EC%9E%90%EB%A3%8C/%EA%B0%9D%EC%B2%B4%20%EC%9D%B8%EC%8B%9D/YoloV8_%EC%8B%A4%EC%8A%B5_%EA%B5%90%EC%9C%A1%EC%9E%90%EB%A3%8C.pdf)

    강의 내용 
     - YOLOv8 학습 소개 
     - 학습 방법 소개 
     - 학습된 모델을 이용한 Inference 방법 소개 
     - 학습한 모델 : Gradio 이용한 GUI 실습 

### 5. [강의자료-5 링크](https://github.com/KangHoyong/D-Jobs_24_02/blob/main/%EA%B0%95%EC%9D%98%EC%9E%90%EB%A3%8C/%EA%B0%9D%EC%B2%B4%20%EC%9D%B8%EC%8B%9D/%EA%B0%9D%EC%B2%B4%EC%9D%B8%EC%8B%9D%20%EA%B5%90%EC%9C%A1%EC%9E%90%EB%A3%8C%2004.pdf)

    강의 내용 
     - Two Stage Detection 소개  


## 실습 코드 

* [yolov8 학습 데이터 링크](https://drive.google.com/file/d/12I2JP9-EfcnLwIhERmAhJ69Sk5teMGIJ/view?usp=sharing) 
```
학습 데이터 소개 
HIT-UAV: A High-altitude Infrared Thermal Dataset : 무인 항공기 고고도 적외선 열 데이터 세트

train image : 2008 
val image : 287
test image : 571

라벨 정의 
  0: Person
  1: Car
  2: Bicycle
  3: OtherVehicle
  4: DontCare

총 라벨 5개 
```