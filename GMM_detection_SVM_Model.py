import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.io import loadmat

################################# Data Loading and Preprocessing #############################



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
X= Training_feature_set.reshape(121935, -1)
y= Training_label_set.ravel()
# Split dataset
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=0)

################################# SVM Training and Evaluation #############################

# Initialize SVM classifier (you can tune parameters)
svm_clf = SVC(kernel='rbf', C=1.0, gamma='scale')

print("Training SVM classifier...")
svm_clf.fit(x_train, y_train)

print("Predicting on test set...")
y_pred = svm_clf.predict(x_test)
Train_y_pred = svm_clf.predict(x_train)

# Accuracy
test_acc = accuracy_score(y_test, y_pred)
Train_acc = accuracy_score(y_train, Train_y_pred)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Train Accuracy: {Train_acc:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=list(range(8)))
print("Confusion Matrix:")
print(cm)

def classification_metrics(cm):
    cm = np.array(cm, dtype=float)
    n_classes = cm.shape[0]
    metrics = {}

    total_samples = cm.sum()
    observed_acc = np.trace(cm) / total_samples
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    expected_acc = np.sum(row_sums * col_sums) / (total_samples ** 2)

    for i in range(n_classes):
        PA = 100*(cm[i, i] / row_sums[i] if row_sums[i] > 0 else 0.0)
        CE = 100*(1 - (cm[i, i] / col_sums[i] if col_sums[i] > 0 else 0.0))
        metrics[f"Class_{i}"] = {"Producer Accuracy": PA, "Commission Error": CE}

    overall_kappa = (observed_acc - expected_acc) / (1 - expected_acc)
    return metrics, overall_kappa

metrics, kappa = classification_metrics(cm)
print("\nClass-wise metrics:")
for cls, vals in metrics.items():
    print(f"{cls}: PA={vals['Producer Accuracy']:.4f}, CE={vals['Commission Error']:.4f}")
print(f"\nOverall Kappa: {kappa:.4f}")
