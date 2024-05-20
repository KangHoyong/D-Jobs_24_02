# 실습 02 : CNN 전체적인 네트워크 구조
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# 장치 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CNN 모델 정의
class CNN(nn.Module) :
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(7*7*32, 10)
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)  # 특성 맵을 벡터로 펼치기
        out = self.fc(out)

        return out

# MNIST 데이터셋 로드
train_dataset = MNIST(root='./data', train=True, transform=ToTensor(), download=True)
test_dataset = MNIST(root='./data', train=False, transform=ToTensor())

# 데이터 로더 생성
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 모델 인스턴스 생성 및 장치로 이동
model = CNN().to(device)

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 설정
num_epochs = 45

# 학습 루프
for epoch in range(num_epochs) :
    model.train()
    running_loss = 0.0

    for images, labels in train_loader :
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_dataset)

    print(f"Epoch [{epoch + 1} / {num_epochs}], Loss : {epoch_loss:.4f}")


# 평가 루프
model.eval()
correct = 0
total = 0

with torch.no_grad() :
    for images, labels in test_loader :
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = 100.0 * correct / total
    print(f"TEST Acc : {acc :.2f}%")

