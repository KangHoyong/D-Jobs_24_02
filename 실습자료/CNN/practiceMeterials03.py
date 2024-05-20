# CNN 실습 03 : 각 에포크에서 합성곱 레이어의 가중치, 완결 연결 레이어의 가중치, 손실 그래프 시각화
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from mpl_toolkits.axes_grid1 import make_axes_locatable

# GPU and CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CNN model
class CNN(nn.Module) :
    def __init__(self):

        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(7 * 7 * 32, out_features=10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

# MNIST dataset loader
train_dataset = MNIST(root='./data', train=True, transform=ToTensor(),download=True)
test_dataset = MNIST(root='./data', train=False, transform=ToTensor())

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

fig, axs = plt.subplots(2, 2, figsize=(10,8))
fig.tight_layout(pad=4.0)
axs = axs.flatten()

# train loop
epoch_loss_list = []

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
    epoch_loss_list.append(epoch_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss : {epoch_loss:.4f}")

    # 1 layer weight
    if epoch == 0 :
        weights = model.conv1.weight.detach().cpu().numpy()
        axs[0].imshow(weights[0, 0], cmap='coolwarm')
        axs[0].set_title('Conv1 Weights')
        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(axs[0].imshow(weights[0,0], cmap='coolwarm'), cax=cax)

    # 2 layer weight
    if epoch == 0 :
        weights = model.conv2.weight.detach().cpu().numpy()
        axs[1].imshow(weights[0, 0], cmap='coolwarm')
        axs[1].set_title('Conv2 Weights')
        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(axs[1].imshow(weights[0,0], cmap='coolwarm'), cax=cax)

    # fc layer weight
    if epoch == 0 :
        weights = model.fc.weight.detach().cpu().numpy()
        axs[2].imshow(weights.T, cmap='coolwarm', aspect='auto')
        axs[2].set_title('FC Weights')
        axs[2].set_xlabel("Input Features")
        axs[2].set_ylabel("Output Units")
        divider = make_axes_locatable(axs[2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(axs[2].imshow(weights.T, cmap='coolwarm', aspect='auto'), cax=cax)

    axs[3].plot(range(epoch+1), epoch_loss_list)
    axs[3].set_title('Training Loss')
    axs[3].set_xlabel('Epoch')
    axs[3].set_ylabel('Loss')

plt.show()

# model eval
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
print(f"Test Accuracy : {acc:.2f}%")
