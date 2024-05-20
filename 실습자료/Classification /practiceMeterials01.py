# pip install seaborn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from datetime import datetime
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# model
class LeNet5(nn.Module) :

    def __init__(self , num_class):
        super(LeNet5, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),

            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=num_class)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs

def get_accuracy(model, data_loader, device) :
    correct_pred = 0
    n = 0

    with torch.no_grad():
        model.eval()
        for x, y_true in data_loader :
            x = x.to(device)
            y_true = y_true.to(device)

            _, y_prob = model(x)
            _, predicted_labels = torch.max(y_prob, 1)

            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum()

        return correct_pred.float() / n

def plot_losses(train_losses, valid_losses) :
    # training loss and valid loss view

    train_losses = np.array(train_losses)
    valid_losses = np.array(valid_losses)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(train_losses, color='blue', label='Training loss')
    ax.plot(valid_losses, color='red', label='validation loss')
    ax.set(title='Loss over epochs',
           xlabel = 'Epoch',
           ylabel =' Loss'
           )
    ax.legend()
    plt.show()

def train(train_loader, model, criterion, optimizer, device) :
    print("Training ...")

    model.train()
    running_loss = 0.0

    for images, labels in train_loader :
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        y_hat, _ = model(images)
        loss = criterion(y_hat, labels)
        running_loss += loss.item() * images.size(0)

        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss

def validate(valid_loader, model, criterion, device) :
    print("Vali ... start")

    model.eval()
    running_loss = 0.0

    for images, labels in valid_loader :
        images = images.to(device)
        labels = labels.to(device)

        y_hat, _ = model(images)
        loss = criterion(y_hat, labels)
        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(val_loader.dataset)

    return model, epoch_loss

def training_loop(model, criterion, optimizer, train_loader, valid_loader, epochs, device, print_every = 1) :

    train_loss_list = []
    valid_loss_list = []

    for epoch in range(0, epochs) :

        # training
        model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device)
        train_loss_list.append(train_loss)

        # validation
        with torch.no_grad() :
            model, valid_loss = validate(valid_loader, model, criterion, device)
            valid_loss_list.append(valid_loss)

        if epoch % print_every == (print_every -1) :
            train_acc = get_accuracy(model, train_loader, device)
            valid_acc = get_accuracy(model, valid_loader, device)

            print(f"{datetime.now().time().replace(microsecond=0)} ---"
                  f"Epoch : {epoch + 1}\t"
                  f"Train loss : {train_loss:.4f}\t"
                  f"Valid loss : {valid_loss:.4f}\t"
                  f"Train acc : {100 * train_acc:.2f}\t"
                  f"Valid acc : {100 * valid_acc:.2f}")

    plot_losses(train_loss_list, valid_loss_list)

    return model, optimizer, (train_loss_list, valid_loss_list)

if __name__ == "__main__" :

    device = "cuda" if torch.cuda.is_available() else 'cpu'

    RANDOM_SEED = 42
    lr = 0.001
    batch_size = 128
    num_epoch = 40
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
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            normalize,
        ]))

    test_dataset = datasets.ImageFolder(
        testdir,
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])
    )
    # data loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # 학습 이미지 샘플 5장 시각화
    # plt.figure(figsize=(15, 10))
    # for i in range(5):
    #     image, label = train_dataset[i]
    #     plt.subplot(1, 5, i + 1)
    #     plt.imshow(image.permute(1, 2, 0))  # 채널 순서 변경 (C, H, W) -> (H, W, C)
    #     plt.title(f'Label: {label}')
    #     plt.axis('off')
    # plt.show()

    torch.manual_seed(RANDOM_SEED)

    model = LeNet5(num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model, optimizer, _ = training_loop(model, criterion, optimizer, train_loader, val_loader, num_epoch, device)

