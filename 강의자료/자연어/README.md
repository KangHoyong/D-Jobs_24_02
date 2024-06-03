# D-Jobs_24_02
D-Jobs 아카데미 2기 자연어 기본 교육 

## D-Jobs 24 2기 강의자료 

## 1. [강의자료-1 링크](https://github.com/KangHoyong/D-Jobs_24_02/blob/main/%EA%B0%95%EC%9D%98%EC%9E%90%EB%A3%8C/%EC%9E%90%EC%97%B0%EC%96%B4/%EC%9E%90%EC%97%B0%EC%96%B4_%EA%B8%B0%EB%B3%B8_%EA%B5%90%EC%9C%A1.pdf)

    강의 내용 
     - 자연어 기본 개념 설명 
     - 택스트 데이터 전처리 실습 
     - 순환 신경망 RNN 소개 및 실습 
     - LSTM 소개 및 실습 
     - seq2seq 소개 
     - 심화 주제 (어텐션 논문 리뷰, 트랜스 포머 소개)

* [RNN 실습 코드](https://github.com/KangHoyong/D-Jobs_24_02/tree/main/%EC%8B%A4%EC%8A%B5%EC%9E%90%EB%A3%8C/%EC%9E%90%EC%97%B0%EC%96%B4/RNN)

RNN 실습 진행 시 오류 사항 공유 
~~~
오류 내용 (데이터 다운로드 후 체크섬에 의해서 데이터 체크 하는 중에 문제 발생)
datasets.utils.info_utils.NonMatchingSplitsSizesError: [{'expected': SplitInfo(name='train', num_bytes=8467781,
num_examples=51094, shard_lengths=None, dataset_name=None), 'recorded': SplitInfo(name='train', num_bytes=21073497,
num_examples=64727, shard_lengths=None, dataset_name='superb')}]

해결 방법
dataset = load_dataset("superb", "ks", ignore_verifications=True)
--> ignore_verifications = True 설정 : 체크섬, 크기 분할 확인을 무시
~~~

* [LSTM 실습 코드](https://github.com/KangHoyong/D-Jobs_24_02/tree/main/%EC%8B%A4%EC%8A%B5%EC%9E%90%EB%A3%8C/%EC%9E%90%EC%97%B0%EC%96%B4/LSTM)

실습 코드 정리 
~~~
데이터 : 네이버 영화 리뷰 데이터 총 200,000개 리뷰로 구성
1. dataprocessing.py : 데이터 프로세싱 코드 
2. tokenized_processing.py : 데이터 프로세싱 완료 -> 데이터 토큰화 처리 코드 
3. word_dict.py : 단어 집합 처리 코드 
4. text_to_sequences.py : 정수 인코딩 코드 
5. lstm_my_model.py : 모델 정의 코드 
6. lstm_train_loop : 학습 및 평가 코드 
7. eval.py : 테스트 코드 
8. main.py : 메인 코드 
~~~

설치 필요 라이브러리
~~~
pip install python-mecab-ko 
~~~

* [LSTM 실습 Dataset : 네이버 영화 리뷰](https://github.com/KangHoyong/D-Jobs_24_02/tree/main/%EC%8B%A4%EC%8A%B5%EC%9E%90%EB%A3%8C/%EC%9E%90%EC%97%B0%EC%96%B4/LSTM/lstm_dataset)