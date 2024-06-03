# LSTM 이용한 네이버 영화 리뷰 분류 - 단어 집합 만들기
from collections import Counter

def word_dict(x_train) :

    word_list = []
    for sent in x_train :
        for word in sent :
            word_list.append(word)

    word_counts = Counter(word_list)

    print("총 단어수 : ", len(word_counts))
    print("훈련 데이터에서의 단어 영화의 등장 횟수 : ", word_counts['영화'])
    print("훈련 데이터에서의 단어 공감의 등장 횟수 : ", word_counts['공감'])
    vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    print("등장 빈도수 상위 10개 단어")
    print(vocab[:10])

    threshold = 3
    total_cnt = len(word_counts) # 단어수
    rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
    total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
    rare_freq = 0 # 등장 빈도수가 threshold 보다 작은 단어의 등장 빈도수의 총합

    # 단어와 빈도수의 쌍을 key와 value로 받는다.
    for key, value in word_counts.items() :
        total_freq = total_freq + value

        # 단어의 등장 빈도수가 threshold보다 작으면
        if (value < threshold) :
            rare_cnt = rare_cnt + 1
            rare_freq = rare_freq + value

    print("단어 집합의 크기 : ", total_cnt)
    print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s' % (threshold - 1, rare_cnt))
    print("단어 집합에서 희귀 단어의 비율 : ", (rare_cnt / total_cnt) * 100)
    print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율 :", (rare_freq / total_freq)*100)

    # 전체 단어 개수 중 빈도수 2이하인 단어 제거
    vocab_size = total_cnt - rare_cnt
    vocab = vocab[:vocab_size]
    print("단어 집합의 크기 : ", len(vocab))

    word_to_index = {}
    word_to_index['<PAD>'] = 0 # padding 문장의 길이를 맞추기 위한 작업
    # 기계가 모르는 단어가 등장하면 그 단어를 단어 집합에 없는 단어란 의미에서 해당 토큰을 UNK라고 한다
    word_to_index['<UNK>'] = 1 # 없는 글자를 처리 하기 위한 작업

    vocab_size = len(word_to_index)
    print('패딩 토큰과 UNK 토큰을 고려한 단어 집합의 크기 : ', vocab_size)
    print('단어 <PAD>와 맵핑되는 정수 : ', word_to_index['<PAD>'])
    print('단어 <UNK>와 맵핑되는 정수 : ', word_to_index['<UNK>'])

    return word_to_index
