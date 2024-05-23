# 실습 05 : 텐서보드 활용 

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )

num_classes = 27
batch_size = 64
num_epochs = 50

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, out_features=num_classes)
model = model.to(device)

# Load data preprocess the data
traindir = "./new_dataset/train/"
valdir = "./new_dataset/val/"
testdir = "./new_dataset/test/"

train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ]))

val_dataset = datasets.ImageFolder(
    valdir,
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ]))

test_dataset = datasets.ImageFolder(
    testdir,
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])
)

# data loader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

writer = SummaryWriter(log_dir='logs')

print("Training start ....")
for epoch in range(num_epochs):
    train_loss = 0.0
    model.train()

    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        outputs = model(image)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # log
        step = epoch * len(train_loader) + batch_idx
        writer.add_scalar("Loss/train", loss.item(), step)

    print(f"Epoch [{epoch +1 } / {num_epochs}] Training loss : {train_loss / len(train_loader)}")

    # Validation
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

    val_acc = 100 * correct / total
    print(f"Validation Acc : {val_acc:.2f}%")

    # log
    writer.add_scalar("Accuracy/val", val_acc, epoch)

# log end
writer.close()
