import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

# ✅ Load preprocessed PCA features
X_train = np.load('features/train_features.npy')
X_test = np.load('features/test_features.npy')
y_train = np.load('features/train_labels.npy')
y_test = np.load('features/test_labels.npy')

# ✅ Train Gaussian Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)

# ✅ Evaluate
y_pred = nb.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro')
rec = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
cm = confusion_matrix(y_test, y_pred)

print("\n📊 Naive Bayes (NB) Results:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("\nConfusion Matrix:\n", cm)

# ✅ Save trained model
joblib.dump(nb, "NB_model.pkl")
print("💾 Saved trained Naive Bayes as NB_model.pkl")
