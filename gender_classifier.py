import torch
import torch.nn as nn
import torchvision.models as models

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import torch.optim as optim
import wandb

class GenderDataset(Dataset):
  def __init__(self, root_dir = 'dataset', transform=None):
    self.root_dir = root_dir
    self.transform = transform
    self.men_dir = os.path.join(root_dir, 'MEN')
    self.women_dir = os.path.join(root_dir, 'WOMAN')
    self.men_images = [os.path.join(self.men_dir, img) for img in os.listdir(self.men_dir)]
    self.women_images = [os.path.join(self.women_dir, img) for img in os.listdir(self.women_dir)]
    self.all_images = self.men_images + self.women_images
    self.labels = [[1, 0]] * len(self.men_images) + [[0, 1]] * len(self.women_images)

  def __len__(self):
    return len(self.all_images)

  def __getitem__(self, idx):
    img_path = self.all_images[idx]
    image = Image.open(img_path).convert('RGB')
    label = torch.tensor(self.labels[idx], dtype=torch.float32)
    
    if self.transform:
      image = self.transform(image)
    
    return image, label

class conv_block(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1):
    super().__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    self.bn = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU()

  def forward(self, x):
    return self.relu(self.bn(self.conv(x)))
  
class GenderClassifier(nn.Module):
  def __init__(self, pretrained = False):
    super().__init__()
    if pretrained:
      resnet = models.resnet18(pretrained = True)
      self.backbone = nn.Sequential(*list(resnet.children())[:-2])
    else:
      self.backbone = nn.Sequential(
        conv_block(3, 64, stride = 2),
        conv_block(64, 128),
        conv_block(128, 256, stride = 2),
        conv_block(256, 512),
        conv_block(512, 1024, stride = 2),
        conv_block(1024, 1024),
        conv_block(1024,2048, stride = 2),
        conv_block(2048, 1024),
        conv_block(1024, 512, stride = 2)
      )
    self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    self.fc = nn.Linear(512, 2)
    self.softmax = nn.Softmax(dim = 1)
  
  def forward(self, x):
    x = self.backbone(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)
    x = self.softmax(x)
    return x

wandb.init(project = 'gender-classifier')

transform = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])
dataset = GenderDataset(transform = transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
trainloader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
valloader = DataLoader(val_dataset, batch_size = 32, shuffle = False)

device = torch.device('mps' if torch.backends.mps.is_built() else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

model = GenderClassifier(pretrained = False).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

num_epochs = 100
best_accuracy = 0
patience = 0
for epoch in range(num_epochs):
  model.train()
  for i, (images, labels) in enumerate(trainloader):
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
      print(f'Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}')
  wandb.log({'Train loss': loss.item()})

  model.eval()
  with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in valloader:
      images, labels = images.to(device), labels.to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)
      _, labels = torch.max(labels.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

    accuracy = correct / total
    if accuracy > best_accuracy:
      best_accuracy = accuracy
    else:
      patience += 1
    if patience > 5:
      break
    print(f'Epoch: {epoch}, Validation Accuracy: {accuracy}')
    wandb.log({'Validation accuracy': accuracy})




    

