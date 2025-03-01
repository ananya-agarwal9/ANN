#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset, Dataset


# In[2]:


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# In[3]:


import torch
import torch.nn as nn

class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # Second convolutional block
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Pooling and dropout layers
        self.pool = nn.MaxPool2d(2, 2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        
    def forward(self, x):
        # Pass through first convolution block
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        
        # Pass through second convolution block
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        
        # Global average pooling
        x = self.global_avg_pool(x)
        
        # Flatten the output and pass through fully connected layers
        x = x.view(-1, 512)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Output layer
        
        return x


# In[4]:


# Hyperparameters
batch_size = 64
learning_rate = 0.001
epochs = 50
pseudo_label_threshold = 0.9  # Confidence threshold for pseudo-labeling


# In[5]:


# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# In[6]:


# Load STL-10 dataset
train_dataset = datasets.STL10(root="./data", split="train", download=True, transform=transform)
test_dataset = datasets.STL10(root="./data", split="test", download=True, transform=transform)
unlabeled_dataset = datasets.STL10(root="./data", split="unlabeled", download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
unlabeled_loader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=False)


# In[ ]:


# Model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImprovedCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Define a learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # Reduce LR every 10 epochs

train_losses = []  # List to store the training losses
train_accuracies = []  # List to store the training accuracies
test_losses = []
test_accuracies = [] 


# In[ ]:


pseudo_labeled_images = False  # Initially set to False

for epoch in range(epochs):
    # Supervised training
    model.train()
    running_loss, correct_train, total_train = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(100 * correct_train / total_train)

    # Fine-tune with pseudo-labeled data
    if pseudo_labeled_images:
        model.train()
        for images, labels in pseudo_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Validation
    model.eval()
    running_test_loss, correct_test, total_test = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
    
    test_losses.append(running_test_loss / len(test_loader))
    test_accuracies.append(100 * correct_test / total_test)

    # Step the scheduler
    scheduler.step()

    print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.2f}%, "
          f"Validation Loss: {test_losses[-1]:.4f}, Val Acc: {test_accuracies[-1]:.2f}%")


# In[ ]:


import matplotlib.pyplot as plt

# Plot Training and Validation Loss vs Epochs
plt.figure(figsize=(12, 6))

# Plot for Loss
plt.subplot(1, 2, 1)
plt.plot(range(epochs), train_losses, label="Training Loss", color='blue')
plt.plot(range(epochs), test_losses, label="Validation Loss", color='red')
plt.title("Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Plot for Accuracy
plt.subplot(1, 2, 2)
plt.plot(range(epochs), train_accuracies, label="Training Accuracy", color='blue')
plt.plot(range(epochs), test_accuracies, label="Validation Accuracy", color='red')
plt.title("Accuracy vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.tight_layout()
plt.show()


# In[ ]:


import pandas as pd

# Create a DataFrame with the training and validation metrics
data = {
    'Epoch': range(1, epochs + 1),
    'Train Loss': train_losses,
    'Train Accuracy': train_accuracies,
    'Val Loss': test_losses,
    'Val Accuracy': test_accuracies
}

df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('training_validation_metrics.csv', index=False)

print("Data saved to 'training_validation_metrics.csv'")


# In[ ]:




