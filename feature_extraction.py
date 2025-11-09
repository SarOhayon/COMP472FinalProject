import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
import numpy as np
import os

# ✅ Step 1: Define transforms (resize for ResNet)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

# ✅ Step 2: Load dataset
train_data = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
test_data = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# ✅ Step 3: Load pretrained ResNet18 and remove final layer
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
feature_extractor.eval()

# ✅ Step 4: Extract features
def extract_features(dataloader):
    features, labels = [], []
    with torch.no_grad():
        for images, lbls in dataloader:
            output = feature_extractor(images)
            output = output.view(output.size(0), -1)  # flatten 512×1
            features.append(output.numpy())
            labels.append(lbls.numpy())
    return np.concatenate(features), np.concatenate(labels)

print("🔹 Extracting training features...")
train_features, train_labels = extract_features(train_loader)
print("🔹 Extracting test features...")
test_features, test_labels = extract_features(test_loader)

print("✅ Feature extraction complete!")
print(f"Train features shape: {train_features.shape}")
print(f"Test features shape:  {test_features.shape}")

# ✅ Step 5: Apply PCA (reduce from 512 → 50)
print("🔹 Applying PCA reduction (512 → 50)...")
pca = PCA(n_components=50)
train_features_pca = pca.fit_transform(train_features)
test_features_pca = pca.transform(test_features)
print("✅ PCA reduction complete!")

# ✅ Step 6: Save as .npy files
os.makedirs('features', exist_ok=True)
np.save('features/train_features.npy', train_features_pca)
np.save('features/test_features.npy', test_features_pca)
np.save('features/train_labels.npy', train_labels)
np.save('features/test_labels.npy', test_labels)
print("💾 Saved features in /features folder!")

