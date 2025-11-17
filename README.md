# COMP472 – Image Classification Project  
### CIFAR-10 Classification using Classical ML & Deep Learning Models

## Authors
- **Lara Louka — 40227840**  
- **Sarah Ohayon — 40209765**

---

This project implements four different machine learning models to classify images from the CIFAR-10 dataset. The models include two classical machine-learning approaches (Naive Bayes, Decision Tree) and two neural-network architectures (MLP, CNN). The project evaluates each model using accuracy, precision, recall, F1-score, confusion matrices, and overall performance.

---

## Models Implemented

### **1. Naive Bayes (GaussianNB)**
- Trained on PCA-reduced ResNet-18 feature vectors  
- Fastest model but lowest performance  
- Struggles with correlated image features  

---

### **2. Decision Tree**
- Trained on PCA-reduced feature vectors  
- Multiple depths were tested  
- Shows overfitting with large depths  

---

### **3. Multi-Layer Perceptron (MLP)**
- **Input:** PCA-reduced features (50 dimensions)  
- **Architecture:**  
  `Linear(50 → 512) → ReLU → Linear(512 → 512) → BatchNorm → ReLU → Linear(512 → 10)`  
- **Epochs:** 20  
- **Optimizer:** SGD (lr = 0.01, momentum = 0.9)  
- **Accuracy:** **77.7%**

---

### **4. Convolutional Neural Network (CNN – VGG11)**
- Input: raw CIFAR-10 images (32×32×3)  
- 5 convolutional blocks + fully connected layers  
- Dropout + BatchNorm to reduce overfitting  
- **Accuracy:** **81.2%** (best model)

---

## Final Results Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Naive Bayes** | 0.7983 | 0.8010 | 0.7983 | 0.7985 |
| **Decision Tree** | 0.6530 | 0.6534 | 0.6530 | 0.6531 |
| **MLP** | 0.7773 | 0.7789 | 0.7773 | 0.7766 |
| **CNN (VGG11)** | **0.8121** | **0.8185** | **0.8121** | **0.8128** |

> **Conclusion:**  
> The CNN achieved the highest performance overall due to its ability to learn hierarchical image features directly from raw pixel space.

---

## How to Run

### **1. Create & activate virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
