# 퍼셉트론 01 실습 : 화학가스 물질 분류 문제
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# CSV 파일 로드
df = pd.read_csv('chemicals_in_wind_tunnel.csv')

# DataFrame의 컬럼 이름 추출
columns = df.columns.tolist()
columns_except_last = columns[:-1]

data = df[[
    'mean_A1', 'mean_A2', 'mean_A3', 'mean_A4', 'mean_A5', 'mean_A6', 'mean_A7', 'mean_A8',
    'mean_B1', 'mean_B2', 'mean_B3', 'mean_B4', 'mean_B5', 'mean_B6', 'mean_B7', 'mean_B8',
    'mean_C1', 'mean_C2', 'mean_C3', 'mean_C4', 'mean_C5', 'mean_C6', 'mean_C7', 'mean_C8',
    'mean_D1', 'mean_D2', 'mean_D3', 'mean_D4', 'mean_D5', 'mean_D6', 'mean_D7', 'mean_D8',
    'mean_E1', 'mean_E2', 'mean_E3', 'mean_E4', 'mean_E5', 'mean_E6', 'mean_E7', 'mean_E8',
    'mean_F1', 'mean_F2', 'mean_F3', 'mean_F4', 'mean_F5', 'mean_F6', 'mean_F7', 'mean_F8',
    'mean_G1', 'mean_G2', 'mean_G3', 'mean_G4', 'mean_G5', 'mean_G6', 'mean_G7', 'mean_G8',
    'mean_H1', 'mean_H2', 'mean_H3', 'mean_H4', 'mean_H5', 'mean_H6', 'mean_H7', 'mean_H8',
    'mean_I1', 'mean_I2', 'mean_I3', 'mean_I4', 'mean_I5', 'mean_I6', 'mean_I7', 'mean_I8',
    'std_A1', 'std_A2', 'std_A3', 'std_A4', 'std_A5', 'std_A6', 'std_A7', 'std_A8',
    'std_B1', 'std_B2', 'std_B3', 'std_B4', 'std_B5', 'std_B6', 'std_B7', 'std_B8',
    'std_C1', 'std_C2', 'std_C3', 'std_C4', 'std_C5', 'std_C6', 'std_C7', 'std_C8',
    'std_D1', 'std_D2', 'std_D3', 'std_D4', 'std_D5', 'std_D6', 'std_D7', 'std_D8',
    'std_E1', 'std_E2', 'std_E3', 'std_E4', 'std_E5', 'std_E6', 'std_E7', 'std_E8',
    'std_F1', 'std_F2', 'std_F3', 'std_F4', 'std_F5', 'std_F6', 'std_F7', 'std_F8',
    'std_G1', 'std_G2', 'std_G3', 'std_G4', 'std_G5', 'std_G6', 'std_G7', 'std_G8',
    'std_H1', 'std_H2', 'std_H3', 'std_H4', 'std_H5', 'std_H6', 'std_H7', 'std_H8',
    'std_I1', 'std_I2', 'std_I3', 'std_I4', 'std_I5', 'std_I6', 'std_I7', 'std_I8',
    'min_A1', 'min_A2', 'min_A3', 'min_A4', 'min_A5', 'min_A6', 'min_A7', 'min_A8',
    'min_B1', 'min_B2', 'min_B3', 'min_B4', 'min_B5', 'min_B6', 'min_B7', 'min_B8',
    'min_C1', 'min_C2', 'min_C3', 'min_C4', 'min_C5', 'min_C6', 'min_C7', 'min_C8',
    'min_D1', 'min_D2', 'min_D3', 'min_D4', 'min_D5', 'min_D6', 'min_D7', 'min_D8',
    'min_E1', 'min_E2', 'min_E3', 'min_E4', 'min_E5', 'min_E6', 'min_E7', 'min_E8',
    'min_F1', 'min_F2', 'min_F3', 'min_F4', 'min_F5', 'min_F6', 'min_F7', 'min_F8',
    'min_G1', 'min_G2', 'min_G3', 'min_G4', 'min_G5', 'min_G6', 'min_G7', 'min_G8',
    'min_H1', 'min_H2', 'min_H3', 'min_H4', 'min_H5', 'min_H6', 'min_H7', 'min_H8',
    'min_I1', 'min_I2', 'min_I3', 'min_I4', 'min_I5', 'min_I6', 'min_I7', 'min_I8',
    'max_A1', 'max_A2', 'max_A3', 'max_A4', 'max_A5', 'max_A6', 'max_A7', 'max_A8',
    'max_B1', 'max_B2', 'max_B3', 'max_B4', 'max_B5', 'max_B6', 'max_B7', 'max_B8',
    'max_C1', 'max_C2', 'max_C3', 'max_C4', 'max_C5', 'max_C6', 'max_C7', 'max_C8',
    'max_D1', 'max_D2', 'max_D3', 'max_D4', 'max_D5', 'max_D6', 'max_D7', 'max_D8',
    'max_E1', 'max_E2', 'max_E3', 'max_E4', 'max_E5', 'max_E6', 'max_E7', 'max_E8',
    'max_F1', 'max_F2', 'max_F3', 'max_F4', 'max_F5', 'max_F6', 'max_F7', 'max_F8',
    'max_G1', 'max_G2', 'max_G3', 'max_G4', 'max_G5', 'max_G6', 'max_G7', 'max_G8',
    'max_H1', 'max_H2', 'max_H3', 'max_H4', 'max_H5', 'max_H6', 'max_H7', 'max_H8',
    'max_I1', 'max_I2', 'max_I3', 'max_I4', 'max_I5', 'max_I6', 'max_I7', 'max_I8'
]].values
labels = df['Chemical']

# 어떤 종류의 값이 있는지 확인 체크
unique_chemicals = df['Chemical'].unique()
"""
총 10개의 클래스를 가지고 있음 
['Acetaldehyde_500' 'Acetone_2500' 'Ammonia_10000' 'Benzene_200'
 'Butanol_100' 'CO_1000' 'CO_4000' 'Ethylene_500' 'Methane_1000'
 'Methanol_200' 'Toluene_200']
"""

chemical_dict = {}
for i, chemical in enumerate(unique_chemicals):
    chemical_dict[chemical] = i

df['Chemical'] = df['Chemical'].map(chemical_dict)

labels = df['Chemical'].values

# 데이터 표준화
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# 데이터를 PyTorch Tensor로 변환
x = torch.tensor(data_scaled, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.long)

# 학습 및 테스트 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print("x_trina 데이터 >> ", len(x_train))
print("x_test 데이터 >> ", len(x_test))

# Dataset 및 DataLoader 생성
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 다층 퍼셉트론 모델 정의
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_size, out_features=512)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=512, out_features=256)
        self.linear3 = nn.Linear(in_features=256, out_features=128)
        self.linear4 = nn.Linear(in_features=128, out_features=64)
        self.linear5 = nn.Linear(in_features=64, out_features=34)
        self.linear6 = nn.Linear(in_features=34, out_features=11)

    def forward(self, x):
        out = self.relu(self.linear1(x))
        out = self.relu(self.linear2(out))
        out = self.relu(self.linear3(out))
        out = self.relu(self.linear4(out))
        out = self.relu(self.linear5(out))
        out = self.linear6(out)
        return out

# 모델 초기화 및 손실 함수, 옵티마이저 정의
model = MLP(input_size=x.shape[1])
criterion = nn.CrossEntropyLoss()  # 평균 제곱 오차 손실 함수
# optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# 모델 학습
num_epochs = 100
for epoch in range(num_epochs):
    model.train()  # 모델을 학습 모드로 설정
    for inputs, targets in train_loader:
        optimizer.zero_grad()  # 기울기 초기화
        outputs = model(inputs)  # 모델 예측
        loss = criterion(outputs, targets)  # 손실 계산
        loss.backward()  # 역전파
        optimizer.step()  # 옵티마이저로 모델 파라미터 업데이트
    print(f"Epoch {epoch + 1}, Train Loss: {loss.item()}")

# 모델 평가
model.eval()  # 모델을 평가 모드로 설정
total_correct = 0
total_samples = 0
with torch.no_grad():  # 그래디언트 계산 비활성화
    for inputs, targets in test_loader:
        outputs = model(inputs)  # 모델 예측
        _, predicted = torch.max(outputs, dim=1)  # 가장 높은 확률을 가진 클래스 선택
        total_samples += targets.size(0)  # 총 샘플 수 업데이트
        total_correct += (predicted == targets).sum().item()  # 맞춘 샘플 수 업데이트

accuracy = total_correct / total_samples  # 정확도 계산
print(f"Test Accuracy: {accuracy}")