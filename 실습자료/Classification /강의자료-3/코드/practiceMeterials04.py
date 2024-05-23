# 전이 학습 실습 -> resnet50 학습 진행 해보기
# 실습 데이터 : 도로 노면 상태 이미지 / 클래스 개수 : 27개

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )

num_classes = 27
batch_size = 64
num_epcohs = 30


model = models.resnet18(pretrained=True)

# Freeze model parameters
# for param in model.parameters() :
#     param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, out_features=num_classes)

# Load data preprocess the data
traindir = "./new_dataset/train/"
valdir = "./new_dataset/val/"
testdir = "./new_dataset/test/"

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

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

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

model = model.to(device)

print("Training start ....")
for epoch in range(num_epcohs) :
    train_loss = 0.0
    model.train()

    for image, label in train_loader :
        image = image.to(device)
        label = label.to(device)
        print(label)
        exit()

        optimizer.zero_grad()

        outputs = model(image)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Epoch [{epoch +1 } / {num_epcohs}] Training loss : {train_loss / len(train_loader)}")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

acc = 100 * correct / total
print(f"Validation Acc : {acc:.2f}%")
