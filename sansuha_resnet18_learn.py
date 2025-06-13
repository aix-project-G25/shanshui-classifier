
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.manifold import TSNE
import seaborn as sns
import cv2
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# === 1. 데이터 로딩 ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
data_dir = "./data"  # 현재 폴더 기준
train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
class_names = train_dataset.classes

# === 2. 모델 준비 ===
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# === 3. 학습 ===
train_losses, val_losses, val_accuracies = [], [], []
best_acc = 0.0
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))

    # 검증
    model.eval()
    correct, total, val_loss = 0, 0, 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(correct / total)
    print(f"Epoch {epoch+1}: Val Accuracy = {val_accuracies[-1]:.4f}")

     # 체크포인트 저장
    if val_accuracies[-1] > best_acc:
        best_acc = val_accuracies[-1]
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accuracy': val_accuracies[-1]
        }, "best_model.pth")
        print(f"✅ Best model saved at epoch {epoch+1} with accuracy {val_accuracies[-1]:.4f}")

# === 4. Loss & Accuracy 시각화 ===
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.plot(val_accuracies, label='Val Accuracy')
plt.legend()
plt.title("Loss & Accuracy")
plt.savefig("loss_accuracy.png")

print("Training complete. Best validation accuracy: {:.4f}".format(best_acc))

