# 실습 04
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchvision.transforms import ToTensor

device = torch.device('cuda' if torch.cuda.is_available() else 'else')

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
        conv1_output = x
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool(out)
        conv2_output = x
        out = out.view(x.size(0), -1)
        out = self.fc(out)
        return out, conv1_output, conv2_output

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

fig, axs = plt.subplots(2, 3, figsize=(15,10))
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

        outputs, conv1_output, conv2_output = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_dataset)
    epoch_loss_list.append(epoch_loss)

    print(f"Epoch [{epoch + 1} / {num_epochs}], Loss : {epoch_loss:.4f}")

    if epoch == 0 :
        # 1 layer weight
        weights = model.conv1.weight.detach().cpu().numpy()
        axs[0].imshow(weights[0, 0], cmap='coolwarm')
        axs[0].set_title('Conv1 Weights')
        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(axs[0].imshow(weights[0,0], cmap='coolwarm'), cax=cax)

        # 2 layer weight
        weights = model.conv2.weight.detach().cpu().numpy()
        axs[1].imshow(weights[0, 0], cmap='coolwarm')
        axs[1].set_title('Conv2 Weights')
        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(axs[1].imshow(weights[0,0], cmap='coolwarm'), cax=cax)

        # 1 Conv
        conv1_output = conv1_output.detach().cpu().numpy()
        axs[2].imshow(conv1_output[0, 0], cmap='coolwarm')
        axs[2].set_title('Conv1 Output')
        divider = make_axes_locatable(axs[2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(axs[2].imshow(conv1_output[0,0], cmap='coolwarm'), cax=cax)

        # 2 Conv
        conv2_output = conv2_output.detach().cpu().numpy()
        axs[3].imshow(conv2_output[0, 0], cmap='coolwarm')
        axs[3].set_title('Conv2 Output')
        divider = make_axes_locatable(axs[3])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(axs[3].imshow(conv2_output[0,0], cmap='coolwarm'), cax=cax)


    # train loas show
    axs[4].plot(range(epoch+1), epoch_loss_list)
    axs[4].set_title("Training Loss")
    axs[4].set_xlabel("Epoch")
    axs[4].set_ylabel("Loss")

plt.show()

