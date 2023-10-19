import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Define data transforms
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create datasets for train and test
train_dataset = ImageFolder('path_to_train_data', transform=data_transforms)
test_dataset = ImageFolder('path_to_test_data', transform=data_transforms)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Define an object detection model (Faster R-CNN for this example)
model = fasterrcnn_resnet50_fpn(pretrained=True)

# Define Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        BCE_loss = nn.BCEWithLogitsLoss()(logits, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss

# Define the optimizer and loss function
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = FocalLoss()

# Training loop (adjust as needed)
for epoch in range(10):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs[0]['scores'], targets.float().view(-1, 1))
        loss.backward()
        optimizer.step()

# Testing loop (adjust as needed)
model.eval()
total_correct = 0
total_samples = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        predicted = (outputs[0]['scores'] > 0.0).float()
        total_correct += (predicted == targets.float().view(-1, 1)).sum().item()
        total_samples += targets.size(0)

accuracy = total_correct / total_samples
print(f"Test Accuracy: {accuracy * 100:.2f}%")
