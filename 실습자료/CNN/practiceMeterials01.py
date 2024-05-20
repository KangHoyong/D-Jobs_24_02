# 재한된 볼츠만 머신 구성 요소 간단 실습
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import torchvision.utils as vutils # 이미지 텐서를 파일로 저장하는 함수입니다. 이 함수는 주로 딥러닝 모델이 생성한 이미지를 저장할 때 사용됩니다.
class RBM(nn.Module) :
    def __init__(self, visible_size, hidden_size):
        super(RBM, self).__init__()
        self.w = nn.Parameter(torch.randn(visible_size, hidden_size))
        self.v_bias = nn.Parameter(torch.randn(visible_size))
        self.h_bias = nn.Parameter(torch.randn(hidden_size))

    def forward(self, x):
        hidden_prob = torch.sigmoid(torch.matmul(x, self.w) + self.h_bias)
        hidden_state = torch.bernoulli(hidden_prob)
        visible_prob = torch.sigmoid(torch.matmul(hidden_state, torch.transpose(self.w, 0, 1)) + self.v_bias)

        return visible_prob, hidden_state

# device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MNIST 데이터셋
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# RBM 모델 인스턴스 생성
visible_size = 784 # MNIST 손글씨 이미지 크기 28x28 = 784
hidden_size = 512
rbm = RBM(visible_size=visible_size, hidden_size=hidden_size).to(device)

# 손실함수, 최적화 알고리즘 정의
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(rbm.parameters(), lr=0.001, momentum=0.9)

# 모델 훈련
num_epochs = 100
for epoch in range(num_epochs) :
    for images, _ in train_loader :
        # 입력 데이터 정규화 및 이진화
        inputs = images.view(-1, visible_size)
        inputs = inputs.to(device)

        visible_prob, _ = rbm(inputs)

        loss = criterion(visible_prob, inputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # log
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss {loss.item():.4f}")

    os.makedirs("./temp_data", exist_ok=True)
    # 가중치 이미지 저장
    vutils.save_image(rbm.w.view(hidden_size, 1, 28, 28), f"./temp_data/weights_epoch_{epoch+1}.jpg", normalize=True)

    # 입력 이미지 및 재구성된 출력 이미지 저장
    inputs_display = inputs.view(-1, 1, 28, 28)
    outputs_display = visible_prob.view(-1, 1, 28, 28)
    comparison = torch.cat([inputs_display, outputs_display], dim=3)
    vutils.save_image(comparison, f"./temp_data/reconstruction_epoch_{epoch +1}.jpg", normalize=True)
