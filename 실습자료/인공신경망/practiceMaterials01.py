# 실습 : 선형 분류 -> 서포트 백터 (PyTroch, sklearn.svm)를 이용하여 SVM 활용한 이미지 분류 실습
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import time

# CSV 파일 로드
df = pd.read_csv('Occupancy_Estimation.csv')

# 날짜 및 시간 열 제거 -> 관계가 없을 것으로 판단 되는데 Data 제거
df = df.drop(['Date', 'Time'], axis=1)

# 상관 행렬 생성
corr_matrix = df.corr()

# 상관 행렬 시각화
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


"""
데이터
    S1_Temp: 센서 1의 온도
    S2_Temp: 센서 2의 온도
    S3_Temp: 센서 3의 온도
    S4_Temp: 센서 4의 온도
    S1_Light: 센서 1의 조명
    S2_Light: 센서 2의 조명
    S3_Light: 센서 3의 조명
    S4_Light: 센서 4의 조명
    S1_Sound: 센서 1의 소리
    S2_Sound: 센서 2의 소리
    S3_Sound: 센서 3의 소리
    S4_Sound: 센서 4의 소리
    S5_CO2: 센서 5의 이산화탄소 농도
    S6_PIR: 센서 6의 PIR(패시브 적외선) 감지
    S7_PIR: 센서 7의 PIR(패시브 적외선) 감지
    
라벨 : 방안에 있는 사람 수 (사람의 수는 0 ~ 3명 사이로 존재)
0명 : 0 
1명 : 1 
2명 : 2 
3명 : 3
"""

sensor_data = df[['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp',
                  'S1_Light', 'S2_Light', 'S3_Light', 'S4_Light',
                  'S1_Sound', 'S2_Sound', 'S3_Sound', 'S4_Sound',
                  'S5_CO2', 'S6_PIR', 'S6_PIR'
                ]].values

labels = df['Room_Occupancy_Count'].values

# 데이터 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(sensor_data)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, labels, test_size=0.2, random_state=777)

# 데이터를 PyTorch Tensor로 변환
x_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(Y_train).long()
x_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(Y_test).long()

# Dataset and Dataloader
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 특징 추출
train_features = []
train_labels = []

for i, data in enumerate(train_loader, 0) :
    inputs, labels = data
    inputs = inputs.view(inputs.shape[0], -1)
    train_features.append(inputs)
    train_labels.append(labels)

train_features = torch.cat(train_features, dim=0)
train_labels = torch.cat(train_labels, dim=0)

test_features = []
test_labels_list = []

for j, test_data in enumerate(test_loader, 0) :
    test_inputs, test_labels = test_data
    test_inputs = test_inputs.view(test_inputs.shape[0], -1)
    test_features.append(test_inputs)
    test_labels_list.append(test_labels)

test_features = torch.cat(test_features, dim=0)
test_labels = torch.cat(test_labels_list, dim=0)

"""
SVM 모델 학습 
SVM 모델은 PyTorch에서 직접 구현되어 있지 않으므로, PyTorch로 작성된 코드에서는 sklearn 모듈을 사용하여 SVM 모델을 구현 필요.
image data 경우는 학습 하는데 시간이 소요 시간이 좀더 걸림.. 지금은 정형 데이터 기준
"""
print("SVM Model Training...")
start_time = time.time()
model = SVC(C=1.0, kernel='rbf', gamma=0.01)
model.fit(train_features.numpy(), train_labels.numpy())
acc = model.score(test_features.numpy(), test_labels.numpy())
end_time = time.time()
print("Accuracy : ", acc)
train_time = end_time - start_time
print("Train 완료 시간 : {:.2f} seconds".format(train_time))


"""
SVM 모델 평가 코드 작성 
1명 정답지 : 
온도 1 >> 24.94 / 온도 2 >> 24.75  / 온도 3 >> 24.56 / 온도 4 >> 25.38 
조명 1 >> 121 / 조명 2 >> 34 / 조명 3 >> 53 / 조명 4 >> 40 
소리 1 >> 0.08 / 소리 2 >> 0.19 / 소리 3 >> 0.06 / 소리 4 >> 0.06
이산화탄소 농도 >> 390
PIR 1 >> 0
PIR 2 >> 0

2명 정답지 : 25.5,25.56,24.88,25.81,155,237,71,55,1.87,0.56,0.23,0.14,500,1,1,2
"""
# 사용자로부터 센서 데이터 입력 받기
sensor_data = []
print("\t")
print("센서 데이터를 입력하세요")
sensor_data.append(float(input("센서 1의 온도를 입력하세요: ")))
sensor_data.append(float(input("센서 2의 온도를 입력하세요: ")))
sensor_data.append(float(input("센서 3의 온도를 입력하세요: ")))
sensor_data.append(float(input("센서 4의 온도를 입력하세요: ")))
sensor_data.append(float(input("센서 1의 조명을 입력하세요: ")))
sensor_data.append(float(input("센서 2의 조명을 입력하세요: ")))
sensor_data.append(float(input("센서 3의 조명을 입력하세요: ")))
sensor_data.append(float(input("센서 4의 조명을 입력하세요: ")))
sensor_data.append(float(input("센서 1의 소리를 입력하세요: ")))
sensor_data.append(float(input("센서 2의 소리를 입력하세요: ")))
sensor_data.append(float(input("센서 3의 소리를 입력하세요: ")))
sensor_data.append(float(input("센서 4의 소리를 입력하세요: ")))
sensor_data.append(float(input("센서 5의 이산화탄소 농도를 입력하세요: ")))
sensor_data.append(int(input("센서 6의 PIR 감지 여부를 입력하세요 (감지: 1, 미감지: 0): ")))
sensor_data.append(int(input("센서 7의 PIR 감지 여부를 입력하세요 (감지: 1, 미감지: 0): ")))

# 입력 받은 데이터 표준화
sensor_data_scaled = scaler.transform([sensor_data])

# SVM 모델을 사용하여 예측 수행
predicted_occupancy_count = model.predict(sensor_data_scaled)

# 결과 출력
print("예측된 방 안의 사람 수:", predicted_occupancy_count[0])
