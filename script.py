import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Normalize, Compose
from sklearn.metrics import accuracy_score


train_dir = './train'
test_dir = './test'
img_dir = './images'

shape_labels = ['circle', 'square', 'triangle']

images = os.listdir(img_dir)
random.shuffle(images)
split = int(0.8 * len(images))
train_images = images[:split]
test_images = images[split:]

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for image in train_images:
    shape = image.split('.')[0]
    src = os.path.join(img_dir, image)
    dst = os.path.join(train_dir, shape, image)
    os.makedirs(os.path.join(train_dir, shape), exist_ok=True)
    os.rename(src, dst)

for image in test_images:
    shape = image.split('.')[0]
    src = os.path.join(img_dir, image)
    dst = os.path.join(test_dir, shape, image)
    os.makedirs(os.path.join(test_dir, shape), exist_ok=True)
    os.rename(src, dst)

train_transforms = Compose([ToTensor(), Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
test_transforms = Compose([ToTensor(), Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

train_dataset = ImageFolder(train_dir, transform=train_transforms)
test_dataset = ImageFolder(test_dir, transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 12 * 12, 64)
        self.fc2 = nn.Linear(64, len(shape_labels))
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 12 * 12)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())


for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

print('Finished training')


true_labels = []
pred_labels = []
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        true_labels += labels.tolist()
        pred_labels += predicted.tolist()

accuracy = accuracy_score(true_labels, pred_labels)
print('Accuracy: %.2f%%' % (accuracy * 100))
