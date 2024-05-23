# 실습 03 : Mixup 실습
# 실습 데이터 : 도로 노면 상태 이미지

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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
            nn.Linear(128 * 26 * 26, 512),
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

model = MyModel(num_classes=27).to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

def mixUp_data(x, y, alpha=0.0) :
    batch_szie = x.size(0) # (batch_size, ch, w, h)
    lam = torch.rand(batch_szie, 1, 1, 1)
    lam = torch.max(lam, 1 - lam)

    # create new alpha
    lam = lam * (1-alpha) + 0.5 * alpha

    mixed_x = lam * x + (1 - lam) * x.flip(dims=[0, 2, 3])
    indices = torch.randperm(batch_szie)
    mixed_y = lam.squeeze() * y + (1 - lam.squeeze()) * y[indices]
    mixed_y = mixed_y.type(torch.long)

    return mixed_x, mixed_y

def plot_images(images, labels, title) :
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle(title, fontsize=16)
    labels = labels.numpy()

    for i, ax in enumerate(axes.flat) :
        image = images[i].squeeze()
        ax.imshow(image, cmap='gray')
        ax.set_title(f"Label : {labels[i]}")
        ax.axis('off')
    plt.show()


num_epochs = 20
train_loss_no_mixup_list = []
train_loss_mixup_list = []

print("training start .......")
end_idx = 0
for epoch in range(num_epochs) :
    train_loss_no_mixup = 0.0
    train_loss_mixup = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader) :
        optimizer.zero_grad()
        # mixup
        mixup_image, mixup_labels = mixUp_data(images, labels, alpha=0.3)
        mixup_image = mixup_image.to(device)
        mixup_labels = mixup_labels.to(device)

        # no mixup
        no_mixup_image = images.to(device)
        no_mixup_label = labels.to(device)

        # Convert mixed image to numpy
        mixed_images = images.cpu().numpy()
        mixed_images = np.transpose(mixed_images, (0,2,3,1))
        mixed_images = np.squeeze(mixed_images)

        if end_idx == 0 :
            plot_images(mixed_images, labels.squeeze(), 'Mixed Image')
            end_idx = 1

        output_mixup = model(mixup_image)
        output_no_mixup = model(no_mixup_image)

        # covert label to 1d tensor
        mixup_labels = torch.squeeze(mixup_labels)

        # loss
        loss_mixup = criterion(output_mixup, mixup_labels)
        loss_no_mixup = criterion(output_no_mixup, no_mixup_label)
        loss_mixup.backward()
        loss_no_mixup.backward()
        optimizer.step()

        train_loss_mixup += loss_mixup.item()
        train_loss_no_mixup += loss_no_mixup.item()

    train_loss_mixup_list.append(train_loss_mixup / len(train_loader))
    train_loss_no_mixup_list.append(train_loss_no_mixup / len(train_loader))
    print(f"epoch : {epoch + 1} / {num_epochs} Mixup Loss : {train_loss_mixup / len(train_loader)}")
    print(f"epoch : {epoch + 1} / {num_epochs} No Mixup Loss : {train_loss_no_mixup / len(train_loader)}")

epochs = range(1, num_epochs + 1)

plt.plot(epochs, train_loss_mixup_list, label="Mixup loss")
plt.plot(epochs, train_loss_no_mixup_list, label='No Mixup loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train Loss Comparison')
plt.legend()
plt.show()

