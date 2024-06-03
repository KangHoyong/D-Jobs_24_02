# LSTM을 이용한 네이버 영화 리뷰 분류 - Data Processing
import pandas as pd
import re
import numpy as np

def data_processing(train_data_path, test_data_path) :

    train_data = pd.read_table(train_data_path)
    test_data = pd.read_table(test_data_path)

    print("train data size : ", len(train_data), "test data size : ", len(test_data))
    # document 열과 label 열의 중복을 제외한 값의 개수
    train_data['document'].nunique(), train_data['label'].unique()
    test_data['document'].nunique(), test_data['label'].unique()

    # document 열의 중복 제거
    train_data.drop_duplicates(subset=['document'], inplace=True)
    test_data.drop_duplicates(subset=['document'], inplace=True)
    print("train data 중복 제거 총 샘플 수 : ", len(train_data))
    print("test data 중복 제거 총 샘플 수 : ", len(test_data))

    # 리뷰 중에 Null 값을 가진 샘플이 있는지 확인
    train_is_null_value = train_data.isnull().values.any()
    test_is_null_value = test_data.isnull().values.any()
    print(train_is_null_value, test_is_null_value)

    train_data = train_data.dropna(how='any')
    test_data = test_data.dropna(how='any')

    # 한글과 공백을 제외하고 모두 제거
    train_data['document'] = train_data['document'].apply(lambda x: re.sub("[^ㄱ-ㅎㅏ-ㅣ가-횡]", "", str(x)))
    test_data['document'] = test_data['document'].apply(lambda x: re.sub("[^ㄱ-ㅎㅏ-ㅣ가-횡]", "", str(x)))

    # 끝에 있는 모든 문제 제거
    train_data['document'] = train_data['document'].apply(lambda x: re.sub(r'\.$', '', x))
    test_data['document'] = test_data['document'].apply(lambda x: re.sub(r'\.$', '', x))

    # 공백을 빈 문자열 대체
    train_data['document'] = train_data['document'].str.replace('^ +', '')
    test_data['document'] = test_data['document'].str.replace('^ +', '')

    # 빈 문자열을 NaN으로 변경
    train_data['document'].replace('', np.nan, inplace=True)
    test_data['document'].replace('', np.nan, inplace=True)

    train_data = train_data.dropna(how='any')
    test_data = test_data.dropna(how='any')
    print(len(train_data), len(test_data))

    return train_data, test_data

# if __name__ == "__main__" :
#     train_data, test_data = data_processing(train_data_path="./lstm_dataset/ratings_train.txt", test_data_path="./lstm_dataset/ratings_test.txt")
#
#     print(train_data, test_data)