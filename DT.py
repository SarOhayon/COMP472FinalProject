import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import joblib

# ✅ Load preprocessed PCA feature data
X_train = np.load('features/train_features.npy')
X_test = np.load('features/test_features.npy')
y_train = np.load('features/train_labels.npy')
y_test = np.load('features/test_labels.npy')

# ✅ Train Decision Tree model
dt = DecisionTreeClassifier(criterion='gini', max_depth=50, random_state=42)
dt.fit(X_train, y_train)

# ✅ Evaluate
y_pred = dt.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro')
rec = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
cm = confusion_matrix(y_test, y_pred)

print("\n📊 Decision Tree (DT) Results:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("\nConfusion Matrix:\n", cm)

# ✅ Experiment: accuracy vs depth
depths = [5, 10, 20, 30, 40, 50, 70]
scores = []
for d in depths:
    temp_dt = DecisionTreeClassifier(max_depth=d, criterion='gini', random_state=42)
    temp_dt.fit(X_train, y_train)
    preds = temp_dt.predict(X_test)
    scores.append(accuracy_score(y_test, preds))

plt.plot(depths, scores, marker='o', color='navy')
plt.title("Decision Tree Accuracy vs Depth")
plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.grid(True)
plt.savefig("dt_depth_curve.png")
plt.close()

# ✅ Save trained model
joblib.dump(dt, "DT_model.pkl")
print("💾 Saved trained Decision Tree as DT_model.pkl")

