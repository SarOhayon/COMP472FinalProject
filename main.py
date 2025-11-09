import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from MLP import MLP
from CNN import VGG11
import joblib

# --- Load preprocessed PCA features (for NB, DT, MLP) ---
X_train = np.load('features/train_features.npy')
X_test = np.load('features/test_features.npy')
y_train = np.load('features/train_labels.npy')
y_test = np.load('features/test_labels.npy')

# ============================================================
# 1️⃣ NAIVE BAYES
# ============================================================
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

acc_nb = accuracy_score(y_test, y_pred_nb)
prec_nb = precision_score(y_test, y_pred_nb, average='macro')
rec_nb = recall_score(y_test, y_pred_nb, average='macro')
f1_nb = f1_score(y_test, y_pred_nb, average='macro')

# ============================================================
# 2️⃣ DECISION TREE
# ============================================================
dt = DecisionTreeClassifier(criterion='gini', max_depth=50, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

acc_dt = accuracy_score(y_test, y_pred_dt)
prec_dt = precision_score(y_test, y_pred_dt, average='macro')
rec_dt = recall_score(y_test, y_pred_dt, average='macro')
f1_dt = f1_score(y_test, y_pred_dt, average='macro')

# ============================================================
# 3️⃣ MULTI-LAYER PERCEPTRON (MLP)
# ============================================================
from torch import nn

mlp = MLP()
mlp.load_state_dict(torch.load("mlp_model.pt"))
mlp.eval()

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
with torch.no_grad():
    preds_mlp = mlp(X_test_tensor)
    y_pred_mlp = torch.argmax(preds_mlp, dim=1).numpy()

acc_mlp = accuracy_score(y_test, y_pred_mlp)
prec_mlp = precision_score(y_test, y_pred_mlp, average='macro')
rec_mlp = recall_score(y_test, y_pred_mlp, average='macro')
f1_mlp = f1_score(y_test, y_pred_mlp, average='macro')

# ============================================================
# 4️⃣ CONVOLUTIONAL NEURAL NETWORK (CNN – VGG11)
# ============================================================
import torchvision
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_data = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

cnn = VGG11().to(device)
cnn.load_state_dict(torch.load("cnn_model.pt", map_location=device))
cnn.eval()

y_true, y_pred_cnn = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = cnn(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        y_pred_cnn.extend(preds)
        y_true.extend(labels.numpy())

acc_cnn = accuracy_score(y_true, y_pred_cnn)
prec_cnn = precision_score(y_true, y_pred_cnn, average='macro')
rec_cnn = recall_score(y_true, y_pred_cnn, average='macro')
f1_cnn = f1_score(y_true, y_pred_cnn, average='macro')

# ============================================================
# 🧾 FINAL RESULTS TABLE
# ============================================================
print("\n📊 FINAL MODEL COMPARISON RESULTS:\n")
print(f"{'Model':<20}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}")
print("-" * 68)
print(f"{'Naive Bayes':<20}{acc_nb:<12.4f}{prec_nb:<12.4f}{rec_nb:<12.4f}{f1_nb:<12.4f}")
print(f"{'Decision Tree':<20}{acc_dt:<12.4f}{prec_dt:<12.4f}{rec_dt:<12.4f}{f1_dt:<12.4f}")
print(f"{'MLP':<20}{acc_mlp:<12.4f}{prec_mlp:<12.4f}{rec_mlp:<12.4f}{f1_mlp:<12.4f}")
print(f"{'CNN (VGG11)':<20}{acc_cnn:<12.4f}{prec_cnn:<12.4f}{rec_cnn:<12.4f}{f1_cnn:<12.4f}")
