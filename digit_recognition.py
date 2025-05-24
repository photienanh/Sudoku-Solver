import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import time
import cv2

class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # 28x28x3 -> 28x28x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)                  # 14x14x32
        )

        self.conv2d_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 14x14x32 -> 14x14x64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)                  # 7x7x64
        )

        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 128),  # 7x7x64 -> 128
            nn.ReLU(),
            nn.Linear(128, 9)  # 128 -> 9 
        )

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    

def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    for epoch in range(num_epochs):
        start_epoch = time.time()
        print(f"Epoch: {epoch + 1}/{num_epochs}")
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = correct_train / total_train

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = correct_val / total_val
        end_epoch = time.time()
        print("Train Loss: {:.4f}, Train Acc: {:.2f}%, Valid Loss: {:.4f}, "
              "Valid Acc: {:.2f}%, Time: {:.2f}s".format(
            avg_train_loss, avg_train_acc * 100,
            avg_val_loss, avg_val_acc * 100, end_epoch - start_epoch))
    return model

def predict_digit(cell_img, model, device):
    img = cv2.resize(cell_img, (28, 28))

    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(tensor)
        pred = output.argmax(dim=1).item() + 1

    return pred

if __name__ == "__main__":
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.GaussianBlur(3, sigma=(0.1, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomRotation(degrees=10),
        transforms.Normalize((0.5,), (0.5,))
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = ImageFolder(root='dataset_digits/train', transform=transform_train)
    val_dataset = ImageFolder(root='dataset_digits/val', transform=transform_val)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model = train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=5)
    torch.save(model.state_dict(), "Model/digit_cnn.pth")