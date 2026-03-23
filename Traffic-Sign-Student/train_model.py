
import pickle                  # to load dataset 
import numpy as np            # numerical operations
import torch                  # deep learning framework
from torch.utils.data import Dataset, DataLoader  # dataset handling
import torchvision.transforms as transforms       # image preprocessing + augmentation
import torchvision.models as models               # pretrained models (ResNet18)
import torch.nn as nn         # neural network layers
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # evaluation
import seaborn as sns         # for confusion matrix visualization
import matplotlib.pyplot as plt

#LOAD DATA
# Loads .p files which contain images and labels
def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['features'], data['labels']

# Load train, validation, and test sets
X_train, y_train = load_data('train.p')
X_val, y_val = load_data('valid.p')
X_test, y_test = load_data('test.p')

print("Original Train shape:", X_train.shape)

# FILTER TO REQUIRED 5 CLASSES
# Original dataset has 43 classes we only need 5

# These are original class IDs from dataset
valid_classes = [14, 33, 34, 35, 40]

# Map them to 0–4 (required for model output)
class_map = {14:0, 33:1, 34:2, 35:3, 40:4}

def filter_data(X, y):
    # Keep only rows where label is in valid_classes
    mask = np.isin(y, valid_classes)
    X = X[mask]
    y = y[mask]

    # Convert original labels to 0–4
    y = np.array([class_map[label] for label in y])

    return X, y

# Apply filtering to all datasets
X_train, y_train = filter_data(X_train, y_train)
X_val, y_val = filter_data(X_val, y_val)
X_test, y_test = filter_data(X_test, y_test)

print("Filtered Train shape:", X_train.shape)
print("Labels:", np.unique(y_train))  # should be [0,1,2,3,4]

# TRANSFORMS 
# Training transformations (include augmentation)
train_transform = transforms.Compose([
    transforms.ToPILImage(),                 # convert numpy toPIL image
    transforms.Resize((32, 32)),             # ensure consistent size
    transforms.RandomRotation(15),           # random rotation (+-15 degrees)
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # slight shifting
    transforms.ColorJitter(brightness=0.2, contrast=0.2),      # lighting variation
    transforms.ToTensor(),           # convert to PyTorch tensor
])

# Validation/Test transformations (NO augmentation)
val_test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# CUSTOM DATASET
# This allows PyTorch to read images + labels properly
class TrafficDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        label = self.y[idx]

        # Apply transformations (augmentation )
        if self.transform:
            img = self.transform(img)

        return img, label

#DATALOADERS 
# Handles batching and shuffling
train_loader = DataLoader(
    TrafficDataset(X_train, y_train, train_transform),
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    TrafficDataset(X_val, y_val, val_test_transform),
    batch_size=64
)

test_loader = DataLoader(
    TrafficDataset(X_test, y_test, val_test_transform),
    batch_size=64
)

print("DataLoader ready")

# MODEL
# Load ResNet18 architecture (no pretrained weights)
model = models.resnet18(weights=None)

# Replace final layer → 5 output classes
model.fc = nn.Linear(model.fc.in_features, 5)

# Use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function (multi-class classification)
criterion = nn.CrossEntropyLoss()

# Optimizer (updates weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Using device:", device)

# TRAINING LOOP
num_epochs = 10

for epoch in range(num_epochs):
    model.train()  # set model to training mode
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()      # clear previous gradients
        outputs = model(images)    # forward pass
        loss = criterion(outputs, labels)  # compute loss

        loss.backward()            # backpropagation
        optimizer.step()           # update weights

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss:.4f}")

# EVALUATION
model.eval()  # switch to evaluation mode
all_preds = []
all_labels = []

with torch.no_grad():  # no gradients needed
    for images, labels in test_loader:
        images = images.to(device)

        outputs = model(images)
        preds = outputs.argmax(dim=1)  # predicted class

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Print metrics
print("\n=== RESULTS ===")
print("Accuracy:", accuracy_score(all_labels, all_preds))
print(classification_report(all_labels, all_preds))

#CONFUSION MATRIX
cm = confusion_matrix(all_labels, all_preds)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()