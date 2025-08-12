import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import scipy.io as sio
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix

################################# Model construction #############################
class SimpleNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten (B, C, H, W) → (B, C*H*W)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        return self.fc4(x)

################################# Model initialization and training #############################
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load features
file_path = 'HMA_AllSamples_Feature_9x9v4.mat'
mat_data_feature = loadmat(file_path)
Training_feature_set = mat_data_feature['Training_feature_set']

# Load labels
file_path = 'HMA_AllSamples_lable.mat'
mat_data_lable = loadmat(file_path)
Training_label_set = mat_data_lable['Training_lable_set'] - 1  # make labels start at 0

# Split
x_train, x_test, y_train, y_test = train_test_split(
    Training_feature_set, Training_label_set, train_size=0.6, random_state=0
)

# Convert to tensors
x_train_t = torch.tensor(x_train[:,4,4,:]).float()
x_test_t = torch.tensor(x_test[:,4,4,:]).float()
y_train_t = torch.tensor(y_train).long()
y_test_t = torch.tensor(y_test).long()

train_dataset = TensorDataset(x_train_t, y_train_t)
val_dataset = TensorDataset(x_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Model, Loss, Optimizer
input_dim = x_train_t.shape[1]                                     #* x_train_t.shape[2] * x_train_t.shape[3]
model = SimpleNN(input_dim=input_dim, num_classes=8).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def classification_metrics(confusion_matrix):
    cm = np.array(confusion_matrix, dtype=float)
    n_classes = cm.shape[0]
    metrics = {}
    total_samples = cm.sum()
    observed_acc = np.trace(cm) / total_samples
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    expected_acc = np.sum(row_sums * col_sums) / (total_samples ** 2)

    for i in range(n_classes):
        PA = cm[i, i] / row_sums[i] if row_sums[i] > 0 else 0.0
        CE = 1 - (cm[i, i] / col_sums[i] if col_sums[i] > 0 else 0.0)
        metrics[f"Class_{i}"] = {
            "Producer Accuracy": PA,
            "Commission Error": CE,
        }
    overall_kappa = (observed_acc - expected_acc) / (1 - expected_acc)
    return metrics, overall_kappa

def train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, num_classes, epochs):
    for epoch in range(1, epochs + 1):
        # ------------------- Training -------------------
        model.train()
        running_loss, running_correct, running_total = 0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device).squeeze()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_correct += outputs.argmax(1).eq(targets).sum().item()
            running_total += targets.size(0)

        train_acc = 100.0 * running_correct / running_total
        avg_loss = running_loss / running_total
        print(f"Epoch [{epoch}/{epochs}] | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%")

    # ------------------- Evaluation -------------------
    model.eval()
    all_outputs, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device).squeeze()
            outputs = model(inputs)
            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())

    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    test_preds = all_outputs.argmax(1)
    test_acc = 100.0 * (test_preds.eq(all_targets)).sum().item() / len(all_targets)

    cm = confusion_matrix(all_targets.numpy(), test_preds.numpy(), labels=list(range(num_classes)))
    print("\nConfusion Matrix:")
    print(cm)

    metrics, kappa = classification_metrics(cm)
    print("\nClass-wise metrics:")
    for cls, vals in metrics.items():
        print(f"{cls}: PA={vals['Producer Accuracy']:.4f}, CE={vals['Commission Error']:.4f}")
    print(f"\nOverall Kappa: {kappa:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

# Run training
train_and_evaluate(model=model, train_loader=train_loader, test_loader=test_loader,
                   optimizer=optimizer, criterion=criterion, num_classes=8, epochs=50)
