# 02 실습 -> 라벨 스무딩 실습
# 실습 데이터 : 도로 노면 상태 이미지

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# class 27

# Label Smoothing Loss Class
class LabelSmoothingLoss(nn.Module) :
    def __init__(self, num_classes, smoothing=0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing # 스무딩 강도
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        smooth_label = one_hot * self.confidence + (1 - one_hot) * self.smoothing / (self.num_classes - 1)
        loss = torch.sum(-smooth_label * torch.log_softmax(pred, dim=1), dim=1)
        return torch.mean(loss)

# My model
class MyModel(nn.Module) :
    def __init__(self, num_classes):
        super(MyModel, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 26 * 26, 512),  # 입력 이미지가 224x224이므로 최종 feature map의 크기는 (128, 26, 26)이 될 것입니다.
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the feature maps
        x = self.fc_layers(x)
        return x


device = "cuda" if torch.cuda.is_available() else 'cpu'

RANDOM_SEED = 42
lr = 0.001
batch_size = 128
num_classes = 27

# Dataset
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
traindir = "./new_dataset/train/"
valdir = "./new_dataset/val/"
testdir = "./new_dataset/test/"

train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize,
    ]))

val_dataset = datasets.ImageFolder(
    valdir,
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize,
    ]))

test_dataset = datasets.ImageFolder(
    testdir,
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize,
    ])
)

# data loader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

criterion = LabelSmoothingLoss(num_classes, smoothing=0.1)

model = MyModel(num_classes=num_classes).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.01)
smoothing_loss = LabelSmoothingLoss(num_classes, smoothing=0.2).to(device)
no_smoothing_loss = torch.nn.CrossEntropyLoss().to(device)

num_epochs = 20
train_losses_no_smoothing_list = []
train_losses_smoothing_list = []

print("Training.....")
for epoch in range(num_epochs) :
    train_losses_smoothing = 0.0
    train_losses_no_smoothing = 0.0

    for images, labels in train_loader :
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        # smoothing
        outputs_smoothing = model(images)
        loss_smoothing = smoothing_loss(outputs_smoothing, labels)
        loss_smoothing.backward()
        optimizer.step()
        train_losses_smoothing += loss_smoothing.item()

        # No smoothing
        outputs_no_smoothing = model(images)
        loss_no_smoothing = no_smoothing_loss(outputs_no_smoothing, labels)
        loss_no_smoothing.backward()
        optimizer.step()
        train_losses_no_smoothing += loss_no_smoothing.item()

    print(f"epoch : {epoch + 1 }/{num_epochs}, loss Smoothing : {train_losses_smoothing / len(train_loader)}")
    print(f"epoch : {epoch + 1 }/{num_epochs}, loss No Smoothing : {train_losses_no_smoothing / len(train_loader)}")

    train_losses_smoothing_list.append(train_losses_smoothing / len(train_loader))
    train_losses_no_smoothing_list.append(train_losses_no_smoothing / len(train_loader))

epochs = range(1, num_epochs + 1)

plt.plot(epochs, train_losses_smoothing_list, label="Smoothing loss")
plt.plot(epochs, train_losses_no_smoothing_list, label='No Smoothing loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train Loss Comparison')
plt.legend()
plt.show()

