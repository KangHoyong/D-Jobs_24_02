# LSTM을 이용한 네이버 영화 리뷰 - 토큰화 및 불용어 처리
from tqdm import tqdm
from mecab import MeCab

def tokenized_processing(train_data, test_data) :

    # 불용어 정의
    stopwords = [
        '도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고',
        '을', '를', '안', '듯', '과', '와', '네', '틀', '듯', '지', '임', '게'
    ]
    x_train = []
    x_test = []

    # 한국어 형태소 분석기 : pip install python-mecab-ko 설치 필요
    mecab = MeCab()
    for sentence in tqdm(train_data['document']) :
        tokenized_sentence = mecab.morphs(sentence) # 토큰화
        stopwords_remove_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
        x_train.append(stopwords_remove_sentence)

    for sentence in tqdm(test_data['document']) :
        tokenized_sentence = mecab.morphs(sentence) # 토큰화
        stopwords_remove_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
        x_test.append(stopwords_remove_sentence)

    return x_train, x_test


