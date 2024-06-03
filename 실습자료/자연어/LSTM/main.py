import torch
import numpy as np
from dataprocessing import data_processing
from tokenized_processing import tokenized_processing
from sklearn.model_selection import train_test_split
from word_dict import word_dict
from text_to_sequences import texts_to_sequences
from lstm_train_loop import lstm_train_loop
from lstm_my_model import TextClassifier
from eval import predict_test

def pad_sequences(sentences, max_len) :
    features = np.zeros((len(sentences), max_len), dtype=int)
    for index, sentences in enumerate(sentences) :
        if len(sentences) != 0 :
            features[index, : len(sentences)] = np.array(sentences)[:max_len]

    print(features)
    return features

def below_threshold_len(max_len, nested_list):
  count = 0
  for sentence in nested_list:
    if(len(sentence) <= max_len):
        count = count + 1
  print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (count / len(nested_list))*100))


if __name__ == "__main__" :
    train_data_path ="./lstm_dataset/ratings_train.txt"
    test_data_path = "./lstm_dataset/ratings_test.txt"

    train_data, test_data = data_processing(train_data_path=train_data_path, test_data_path=test_data_path)

    x_train, y_train = tokenized_processing(train_data, test_data)

    y_train = np.array(train_data['label'])
    y_test = np.array(test_data['label'])

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=0, stratify=y_train)
    # 학습 데이터 비율 체크
    print('--------학습 데이터의 비율-----------')
    print(f'부정 리뷰 = {round(np.sum(y_train == 0) / len(y_train) * 100, 3)}%')
    print(f'긍정 리뷰 = {round(np.count_nonzero(y_train) / len(y_train) * 100, 3)}%')
    print('--------검증 데이터의 비율-----------')
    print(f'부정 리뷰 = {round(np.sum(y_valid == 0) / len(y_valid) * 100, 3)}%')
    print(f'긍정 리뷰 = {round(np.count_nonzero(y_valid) / len(y_valid) * 100, 3)}%')
    print('--------테스트 데이터의 비율-----------')
    print(f'부정 리뷰 = {round(np.sum(y_test == 0) / len(y_test) * 100, 3)}%')
    print(f'긍정 리뷰 = {round(np.count_nonzero(y_test) / len(y_test) * 100, 3)}%')

    word_to_index = word_dict(x_train)
    print("main word to inedx", word_to_index)

    encoded_x_train = texts_to_sequences(tokenized_x_data=x_train, word_to_index=word_to_index)
    encoded_x_valid = texts_to_sequences(tokenized_x_data=x_valid, word_to_index=word_to_index)

    # 리뷰의 최대 길이, 평균 길이
    print('리뷰의 최대 길이 : ', max(len(review) for review in encoded_x_train)) # 74
    print('리뷰의 평균 길이 : ', sum(map(len, encoded_x_train))/len(encoded_x_train)) # 12.0

    # 학습하기 위해서는 학습 데이터 길이를 지정할 필요가 있음 이걸 체크 하기 위한 함수
    # 샘플 비율을 보고 체크 30 -> 92% / 35 -> 94% / 40 -> 96%
    below_threshold_len(max_len=40, nested_list=x_train)

    # LSTM 학습 -> 고정 크기 사이즈가 필요
    # 예) 25 나머지 15에 대해서는 0으로 채워짐
    max_len = 40
    padded_x_train = pad_sequences(encoded_x_train, max_len=max_len)
    padded_x_valid = pad_sequences(encoded_x_valid, max_len=max_len)

    print('훈련 데이터 크기 : ', padded_x_train.shape)
    print('검증 데이터 크기 : ', padded_x_valid.shape)

    print('첫번째 샘플의 길이 : ', len(padded_x_train[0]))
    print('첫번째 샘플 : ', padded_x_train[0])

    """
    첫번째 샘플의 길이 :  40
    첫번째 샘플 :  [1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    """

    # train loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TextClassifier(vocab_size=100, embedding_dim=500, hidden_dim=128, output_dim=2).to(device)
    # lstm_train_loop(y_train=y_train, y_valid=y_valid,padded_x_train=padded_x_train, padded_x_valid=padded_x_valid, num_epoch=100, model=model, device=device)

    # 테스트 입력값을 통한 모델 인퍼런스
    index_to_tag = {0 : '부정', 1 : '긍정'}
    checkpoint_path = "./best_model_checkpoint.pth"
    output_result = predict_test(text="이 영화 재미 없어....",model=model, word_to_index=word_to_index,index_to_tag=index_to_tag, checkpoint_path=checkpoint_path)
    print("모델 예측 결과 ", output_result)
