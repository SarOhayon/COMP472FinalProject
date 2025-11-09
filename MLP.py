import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# ✅ Load preprocessed data
X_train = np.load('features/train_features.npy')
X_test = np.load('features/test_features.npy')
y_train = np.load('features/train_labels.npy')
y_test = np.load('features/test_labels.npy')

# Convert to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# ✅ Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(50, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.model(x)

# ✅ Initialize model, loss, and optimizer
model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# ✅ Training loop
epochs = 20
train_losses = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss.item():.4f}")

# ✅ Plot training loss
plt.plot(train_losses)
plt.title("MLP Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("mlp_loss.png")   # save instead of showing
plt.close()


# ✅ Evaluate on test set

model.eval()
y_pred_list = []
batch_size = 1000

with torch.no_grad():
    for i in range(0, len(X_test), batch_size):
        batch = X_test[i:i+batch_size]
        preds = model(batch)
        y_pred_list.append(torch.argmax(preds, dim=1))

y_pred = torch.cat(y_pred_list)


# ✅ Compute metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro')
rec = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
cm = confusion_matrix(y_test, y_pred)

print("\n📊 Evaluation Results:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("\nConfusion Matrix:\n", cm)

# ✅ Save model
torch.save(model.state_dict(), "mlp_model.pt")
print("\n💾 Saved trained model as mlp_model.pt")
