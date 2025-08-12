import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.ensemble import RandomForestClassifier


file_path = 'HMA_AllSamples_Feature.mat'
mat_data_feature = loadmat(file_path)
# # List all variables
# # Access the variable
Training_feature_set = mat_data_feature['Training_feature_set']
Training_feature_set = Training_feature_set[:,4,4,:]

file_path = 'HMA_AllSamples_lable.mat'
mat_data_lable = loadmat(file_path)
# List all variables
# Access the variable
Training_label_set = mat_data_lable['Training_lable_set'] - 1
# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    Training_feature_set, Training_label_set, train_size=0.6, random_state=0)

# Flatten images for Random Forest
x_train_rf = x_train.reshape(x_train.shape[0], -1)
x_test_rf = x_test.reshape(x_test.shape[0], -1)
y_train_rf = y_train.ravel()
y_test_rf = y_test.ravel()

# ------------------------ Random Forest Model ------------------------
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=0,
    n_jobs=-1
)

print("Training Random Forest...")
rf_model.fit(x_train_rf, y_train_rf)

# Predictions
y_pred = rf_model.predict(x_test_rf)

# ------------------------ Metrics ------------------------
def classification_metrics(cm):
    cm = np.array(cm, dtype=float)
    n_classes = cm.shape[0]
    metrics = {}
    total_samples = cm.sum()

    observed_acc = (np.trace(cm) / total_samples)
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    expected_acc = np.sum(row_sums * col_sums) / (total_samples ** 2)

    for i in range(n_classes):
        PA = 100*(cm[i, i] / row_sums[i] if row_sums[i] > 0 else 0.0)
        CE = 100*(1 - (cm[i, i] / col_sums[i] if col_sums[i] > 0 else 0.0))
        metrics[f"Class_{i}"] = {"Producer Accuracy": PA, "Commission Error": CE}

    overall_kappa = (observed_acc - expected_acc) / (1 - expected_acc)
    return metrics, observed_acc, overall_kappa

cm = confusion_matrix(y_test_rf, y_pred, labels=list(range(8)))
metrics, overall_acc, kappa = classification_metrics(cm)

print("\nConfusion Matrix:")
print(cm)
print(f"\nOverall Accuracy: {overall_acc:.4f}")
print(f"Kappa Coefficient: {kappa:.4f}")
print("\nClass-wise metrics:")
for cls, vals in metrics.items():
    print(f"{cls}: PA={vals['Producer Accuracy']:.4f}, CE={vals['Commission Error']:.4f}")
y_train_pred = rf_model .predict(x_train_rf)
y_test_pred = rf_model .predict(x_test_rf)
train_accuracy = 100*(accuracy_score(y_train, y_train_pred))
test_accuracy = 100*(accuracy_score(y_test, y_test_pred))
print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
