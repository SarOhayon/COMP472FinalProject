from sklearn.decomposition import PCA
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Subset, DataLoader
import torch.nn as nn
import torch
from DecisionTree import DecisionTree

from NaiveBayes import GaussianNaiveBayes
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB

# Telling the model what size to covert images to and normalizing them
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
# loading CIFAR10 dataset. Loading the training and test sets from this dataset
training_set = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)
print(training_set)
print(test_set)

# create a subset from a datset with num_per_class items for each class


def subset_by_class(dataset, num_per_class):
    targets = np.array(dataset.targets)
    indices = []
    for cls in range(len(dataset.classes)):
        cls_idx = np.where(targets == cls)[0][:num_per_class]
        indices.extend(cls_idx)
    return Subset(dataset, indices)


# subsetting the both datasets as specifed in Assignment
training_subset = subset_by_class(training_set, 500)
test_subset = subset_by_class(test_set, 100)

# splits it into batches of 64 cuz it makes it faster
# training:
training_loader = DataLoader(
    training_subset, batch_size=64, shuffle=False, num_workers=0)
# test:
test_loader = DataLoader(
    test_subset, batch_size=64, shuffle=False, num_workers=0)

# Load the resnet18 pre-trianed model
# Remove the last classifcation layer so I can do the classification myself
resnet18 = torchvision.models.resnet18(pretrained=True)
resnet18 = nn.Sequential(*list(resnet18.children())[:-1])
# prepares the model
resnet18.eval()
# puts the model on the cpu (i dont have gpu)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
resnet18.to(device)

# loop to extract features from our images (512x1 vector)


def extract_features(data_loader):
    features = []
    labels = []
    with torch.no_grad():
      # looping over our dataset
        for images, targets in data_loader:
            images = images.to(device)
            # passing the images thru the model -> result should be an array of features
            feats = resnet18(images).squeeze()
            features.append(feats)
            labels.append(targets)
    # concatinate all the arrays so we only have one to search thru
    return torch.cat(features), torch.cat(labels)


# extract features
training_features, training_labels = extract_features(training_loader)
test_features, test_labels = extract_features(test_loader)

pca = PCA(n_components=50)
training_features = pca.fit_transform(training_features.cpu().numpy())
test_features = pca.transform(test_features.cpu().numpy())
y_true = test_labels.cpu().numpy() if hasattr(
    test_labels, "cpu") else test_labels

gnb = GaussianNaiveBayes()
gnb.fit(training_features, training_labels)

y_pred = gnb.predict(test_features)
accuracy = np.mean(y_pred == test_labels.cpu().numpy())
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print("Gaussian Naive Bayes")
print(f"  Accuracy : {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall   : {recall:.4f}")
print(f"  F1-score : {f1:.4f}")
print()

gnb = GaussianNB()
gnb.fit(training_features, training_labels)
y_pred = gnb.predict(test_features)
accuracy = accuracy_score(test_labels, y_pred)
precision = precision_score(test_labels, y_pred, average='weighted')
recall = recall_score(test_labels, y_pred, average='weighted')
f1 = f1_score(test_labels, y_pred, average='weighted')

print("Gaussian Naive Bayes Scikit-Learn:")
print(f"  Accuracy : {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall   : {recall:.4f}")
print(f"  F1-score : {f1:.4f}")
print()

for depth in [100, 50, 25, 10, 5, 2, 1]:

    tree = DecisionTree(max_depth=depth)
    tree.fit(training_features, training_labels)
    y_pred = tree.predict(test_features)
    accuracy = np.mean(y_pred == test_labels.cpu().numpy())
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"Decision Tree (MaxDepth={depth})")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1-score : {f1:.4f}")

    tree = DecisionTreeClassifier(criterion="gini", max_depth=depth)
    tree.fit(training_features, training_labels)
    y_pred = tree.predict(test_features)
    accuracy = accuracy_score(test_labels, y_pred)
    precision = precision_score(test_labels, y_pred, average='weighted')
    recall = recall_score(test_labels, y_pred, average='weighted')
    f1 = f1_score(test_labels, y_pred, average='weighted')

    print(f"Decision Tree (MaxDepth={depth}) Scikit-Learn:")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1-score : {f1:.4f}")
