import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms, datasets
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.manifold import TSNE
import seaborn as sns
from PIL import Image
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === 데이터 로드 (val dataset만) ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
data_dir = "./data"
val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
class_names = val_dataset.classes

# === 모델 로딩 ===
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
checkpoint = torch.load("best_model.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# === 1. 혼동 행렬 + 분류 리포트 ===
all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix_loaded.png")
plt.close()

print(classification_report(all_labels, all_preds, target_names=class_names))

# === 2. 예측 시각화 ===
def visualize_predictions(correct=True, n=8):
    shown = 0
    plt.figure(figsize=(16, 8))
    with torch.no_grad():
        for i, (img, label) in enumerate(val_dataset):
            input_tensor = img.unsqueeze(0).to(device)
            output = model(input_tensor)
            _, pred = torch.max(output, 1)
            if (pred.item() == label) == correct:
                img_disp = img.permute(1, 2, 0).numpy()
                img_disp = img_disp * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
                img_disp = np.clip(img_disp, 0, 1)
                plt.subplot(2, 4, shown + 1)
                plt.imshow(img_disp)
                plt.title(f"GT: {class_names[label]}, Pred: {class_names[pred.item()]}")
                plt.axis('off')
                shown += 1
                if shown == n:
                    break
    plt.suptitle("Correct Predictions" if correct else "Wrong Predictions")
    plt.tight_layout()
    plt.savefig("correct_preds_loaded.png" if correct else "wrong_preds_loaded.png")
    plt.close()

visualize_predictions(correct=True)
visualize_predictions(correct=False)

# === 3. Grad-CAM 시각화 ===
def generate_gradcam_from_image(model, image_path, save_path="gradcam_result.png"):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)

    feature_maps = []
    gradients = []

    def forward_hook(module, input, output):
        feature_maps.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    handle_fw = model.layer4.register_forward_hook(forward_hook)
    handle_bw = model.layer4.register_full_backward_hook(backward_hook)

    output = model(img_tensor)
    pred_class_idx = output.argmax(dim=1).item()
    pred_class_name = class_names[pred_class_idx]

    model.zero_grad()
    class_loss = output[0, pred_class_idx]
    class_loss.backward()

    grads = gradients[0].squeeze(0).cpu().detach().numpy()
    fmap = feature_maps[0].squeeze(0).cpu().detach().numpy()
    weights = np.mean(grads, axis=(1, 2))
    cam = np.sum(weights[:, None, None] * fmap, axis=0)
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    cam = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    img_disp = np.array(image.resize((224, 224)))
    overlay = cv2.addWeighted(img_disp, 0.6, heatmap, 0.4, 0)

    plt.imshow(overlay[..., ::-1])  # BGR → RGB
    plt.title(f"Predicted: {pred_class_name}")
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()

    handle_fw.remove()
    handle_bw.remove()

def get_random_image_path(from_dir):
    image_files = os.listdir(from_dir)
    return os.path.join(from_dir, np.random.choice(image_files))

image1_path = get_random_image_path(from_dir=os.path.join(data_dir, "val", class_names[0]))
generate_gradcam_from_image(model, image1_path, save_path=f"gradcam_result_{class_names[0]}.png")
image2_path = get_random_image_path(from_dir=os.path.join(data_dir, "val", class_names[1]))
generate_gradcam_from_image(model, image2_path, save_path=f"gradcam_result_{class_names[1]}.png")

# === 4. t-SNE 시각화 ===
features, labels = [], []
with torch.no_grad():
    for img, label in val_dataset:
        img = img.unsqueeze(0).to(device)
        feat = model.avgpool(model.layer4(model.layer3(model.layer2(model.layer1(model.relu(model.bn1(model.conv1(img))))))))
        feat = torch.flatten(feat, 1)
        features.append(feat.cpu().numpy())
        labels.append(label)
features = np.concatenate(features)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
reduced = tsne.fit_transform(features)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=[class_names[l] for l in labels], palette="Set1")
plt.title("t-SNE of ResNet18 Features (Loaded Model)")
plt.savefig("tsne_loaded.png")
plt.close()
