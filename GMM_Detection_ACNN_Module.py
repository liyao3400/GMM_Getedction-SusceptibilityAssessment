import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import scipy.io as sio
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from scipy.io import loadmat
import numpy as np
from scipy.io import savemat
from sklearn.metrics import confusion_matrix

#################################Model construction#############################
# Channel Attention
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return torch.sigmoid(avg_out + max_out)

# Spatial Attention
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.conv(x_cat))

# CBAM Block
class CBAM(nn.Module):
    def __init__(self, in_channels):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# Deep CNN with CBAM
class DeepCNNWithAttention(nn.Module):
    def __init__(self, num_classes=8):
        super(DeepCNNWithAttention, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(9, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            CBAM(32),
            nn.MaxPool2d(2)  # 32->16
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            CBAM(64),
            nn.MaxPool2d(2)  # 16->8
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            CBAM(128),
            nn.MaxPool2d(2)  # 8->4
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            CBAM(256),
            nn.AdaptiveAvgPool2d((1, 1))  # Output: 512 x 1 x 1
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

#################################Model initialization and training#############################
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


batch_size = 128
epochs = 50
data_augmentation = False
num_classes = 8
lr = 0.001
subtract_pixel_mean = True  # Subtracting pixel mean improves accuracy
base_model = 'resnet20'
# Choose what attention_module to use: cbam_block / se_block / None
attention_module = 'cbam_block'
model_type = base_model if attention_module==None else base_model+'_'+attention_module


file_path = 'HMA_AllSamples_Feature.mat'
with h5py.File(file_path, 'r') as f:
    # List all variables in the .mat file
    print("Keys:", list(f.keys()))
    # Suppose the variable name is 'data'
    Training_feature_set = f['Training_feature_set'][:].transpose(3,1,2,0)

file_path = 'HMA_AllSamples_lable.mat'
mat_data_lable = loadmat(file_path)
# List all variables
# Access the variable
Training_label_set = mat_data_lable['Training_lable_set'] - 1

# # split
x_train, x_test, y_train, y_test = train_test_split(Training_feature_set, Training_label_set, train_size = 0.6, random_state = 0)

# Convert to tensors and permute to (C, H, W)
x_train_t = torch.tensor(x_train).permute(0, 3, 2,1).float()
x_test_t = torch.tensor(x_test).permute(0, 3, 2,1).float()
y_train_t = torch.tensor(y_train).long()
y_test_t = torch.tensor(y_test).long()

train_dataset = TensorDataset(x_train_t, y_train_t)
val_dataset = TensorDataset(x_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)



# Model, Loss, Optimizer
model = DeepCNNWithAttention(num_classes=8).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)



def classification_metrics(confusion_matrix):
    """
    Calculate Producer's Accuracy (PA), Commission Error (CE), and Kappa for each category.

    Parameters:
        confusion_matrix (numpy.ndarray): Square confusion matrix (n_classes x n_classes)

    Returns:
        dict: Dictionary with PA, CE, and Kappa for each class
    """

    # Ensure matrix is numpy array
    cm = np.array(confusion_matrix, dtype=float)
    n_classes = cm.shape[0]
    metrics = {}

    # Total samples
    total_samples = cm.sum()

    # Observed accuracy
    observed_acc = np.trace(cm) / total_samples

    # Expected accuracy (for kappa)
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    expected_acc = np.sum(row_sums * col_sums) / (total_samples ** 2)

    for i in range(n_classes):
        # Producer's Accuracy (Recall for class i)
        PA = 100*(cm[i, i] / row_sums[i] if row_sums[i] > 0 else 0.0)

        # Commission Error (1 - Precision)
        CE = 100*(1 - (cm[i, i] / col_sums[i] if col_sums[i] > 0 else 0.0))


        metrics[f"Class_{i}"] = {
            "Producer Accuracy": PA,
            "Commission Error": CE,

        }

    # Overall kappa
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

        # Print Results
        print(f"Epoch [{epoch}/{epochs}]")
        print(f"Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")

    # ------------------- Evaluation -------------------
    model.eval()
    test_correct, test_total = 0, 0
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

    Train_all_outputs, Train_all_targets = [], []
    with torch.no_grad():
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device).squeeze()
            outputs = model(inputs)
            Train_all_outputs.append(outputs.cpu())
            Train_all_targets.append(targets.cpu())

    Train_all_outputs = torch.cat(Train_all_outputs)
    Train_all_targets = torch.cat(Train_all_targets)
    Train_preds = Train_all_outputs.argmax(1)
    Train_acc = 100.0 * (Train_preds.eq(Train_all_targets)).sum().item() / len(Train_all_targets)

    # Compute FAR and FRR
    #Acc, FAR, FRR = compute_far_frr(all_outputs, all_targets, num_classes)

    # Confusion Matrix
    cm = confusion_matrix(all_targets.numpy(), test_preds.numpy(), labels=list(range(num_classes)))
    print("\nConfusion Matrix:")
    print(cm)

    # PA, CE, Kappa
    metrics, kappa = classification_metrics(cm)
    print("\nClass-wise metrics:")
    for cls, vals in metrics.items():
        print(f"{cls}: PA={vals['Producer Accuracy']:.4f}, CE={vals['Commission Error']:.4f}")
    #print(f"\nOverall Kappa: {kappa:.4f}")
    print(f"\nTest Overall Accuracy: {test_acc:.4f}")
    print(f"\nTrain Overall Accuracy: {Train_acc:.4f}")

train_and_evaluate(model=model, train_loader=train_loader, test_loader=test_loader, optimizer=optimizer, criterion=criterion, num_classes=8, epochs=5)


####################################################Output All result#################################

# All_predicted = np.zeros((Image_label_set.shape[0] - 2 * half_rows,
#                           Image_label_set.shape[1] - 2 * half_clos),
#                          dtype=np.int64)
#
# for i in range(half_rows,Image_label_set.shape[0] - half_rows):
#     Row_feature_list = []
#
#     for j in range(half_clos, Image_label_set.shape[1] - half_clos):
#         patch = Nor_All_Feature[i - half_rows:i + half_rows,
#                 j - half_clos:j + half_clos, :]  # (32,32,7)
#         Row_feature_list.append(patch)
#
#     # Convert row patches to tensor: (N, C, H, W)
#     Row_feature_set = torch.tensor(np.array(Row_feature_list)).permute(0, 3, 1, 2).float()
#     Row_feature_dataset = TensorDataset(Row_feature_set)  # features only
#     Row_feature_loader = DataLoader(Row_feature_dataset, batch_size=64, shuffle=False)  # shuffle=False!
#  
#     # Predict for this row
#     row_preds = []
#     with torch.no_grad():
#         for batch in Row_feature_loader:
#             inputs = batch[0].to(device)
#             outputs = model(inputs)
#             _, predicted = outputs.max(1)
#             row_preds.append(predicted.cpu().numpy())
#
#     row_preds = np.concatenate(row_preds)
#     All_predicted[i - half_rows, :] = row_preds  # Correct row index
#
#     print(f"Processing row {i}/{Image_label_set.shape[0] - half_rows}")
#
# # Save to .mat file
# sio.savemat('Predict_label.mat',
#             mdict={'predict_label': All_predicted})
# print("Full prediction map saved!")



