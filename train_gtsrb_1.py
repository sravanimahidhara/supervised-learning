#import required packages

import os
import csv
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

#directory paths for the data sets
TRAIN_DIR = "/Users/sravani/project/data/GTSRB/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images"
TEST_IMAGES_DIR = "/Users/sravani/project/data/GTSRB/GTSRB_Final_Test_Images/GTSRB/Final_Test/Images"
GT_CSV_PATH = "/Users/sravani/project/data/GTSRB/GTSRB_Final_Test_GT/GTSRB/GT-final_test.csv"


#configurations
IMG_SIZE = 32
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3

#data transforms
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])


#test dataset
class GTSRBTestDataset(Dataset):
    def __init__(self, images_dir, gt_csv, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.samples = []

        with open(gt_csv, newline="") as f:
            reader = csv.reader(f, delimiter=";")
            next(reader)
            for row in reader:
                self.samples.append((row[0], int(row[7])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label


#convolutional neural network
class TrafficSignNet(nn.Module):
    def __init__(self, num_classes=43):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


#train and evaluate functions
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            correct += outputs.argmax(1).eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


#plots
def plot_learning_curves(train_losses, test_losses, train_accs, test_accs):
    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(epochs, train_accs, label="Train Accuracy")
    plt.plot(epochs, test_accs, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.show()


#main function
def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    test_dataset = GTSRBTestDataset(TEST_IMAGES_DIR, GT_CSV_PATH, test_transform)

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=0)

    model = TrafficSignNet(len(train_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    for epoch in range(NUM_EPOCHS):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)

        train_losses.append(tr_loss)
        test_losses.append(te_loss)
        train_accs.append(tr_acc)
        test_accs.append(te_acc)

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
            f"Test Loss: {te_loss:.4f} Acc: {te_acc:.4f}"
        )

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/gtsrb_cnn.pth")
    print("Model saved to models/gtsrb_cnn.pth")

    plot_learning_curves(train_losses, test_losses, train_accs, test_accs)


if __name__ == "__main__":
    main()

