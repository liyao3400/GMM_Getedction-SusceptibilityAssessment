import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
import math

# ===== 1. Load the data from Excel =====
file_path = "samples.xls"
sheet_name = "Sheet1"
data = pd.read_excel(file_path, sheet_name=sheet_name,encoding='utf-8').values

# ===== 2. Positive and negative sample selection =====
positive_data = data[0:426, :]
idx_neg = np.random.randint(434, 3296, size=426)
negative_data = data[idx_neg, :]
all_data = np.vstack((positive_data, negative_data))

# ===== 3. Normalization based on factor ranges =====

all_data[:, 1] = all_data[:, 1]/403;      
all_data[:, 2] = all_data[:, 2]/7820;      
all_data[:, 3] = all_data[:, 3]/78.5;  
all_data[:, 4] = all_data[:, 4]/74;     
all_data[:, 5] = np.log10(all_data[:, 5]+1)/np.log10(1078305758);   
all_data[:, 6] = (all_data[:, 6]+10)/20;  
all_data[:, 7] = (all_data[:, 7]+469)/1694;    
all_data[:, 8] = np.log10(all_data[:, 8] * 1e6+1) / np.log10(11768)
all_data[:, 9] = np.log10(all_data[:, 9]+1)/np.log10(60052788);   
all_data[:, 10] = np.log10(all_data[:, 10]+1)/np.log10(1701824);   
all_data[:, 11] = all_data[:, 11]/2205;    
all_data[:, 12] = (all_data[:, 12]+11.82)/33.47;
all_data[:, 13] = (all_data[:, 13])/0.073;    

# ===== 4. Train-test split =====
X = all_data[:, 1:14]   # features
y_raw = all_data[:, 14] # labels

# One-hot encode labels
num_classes = int(y_raw.max()) + 1
y = np.zeros((len(y_raw), num_classes))
for i, lbl in enumerate(y_raw.astype(int)):
    if lbl > 0:
        y[i, lbl] = 1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)

# ===== 5. Build Sparse Stacked Autoencoder =====
# L1 regularization encourages sparsity
input_dim = X_train.shape[1]

# First autoencoder

input_layer = layers.Input(shape=(input_dim,))
encoded1 = layers.Dense(64, activation='relu',
                        activity_regularizer=regularizers.l1(1e-5))(input_layer)
encoded2 = layers.Dense(64, activation='relu',
                        activity_regularizer=regularizers.l1(1e-5))(encoded1)
encoded3 = layers.Dense(64, activation='relu',
                        activity_regularizer=regularizers.l1(1e-5))(encoded2)
decoded3 = layers.Dense(64, activation='relu')(encoded3)
decoded2 = layers.Dense(64, activation='relu')(decoded3)
decoded1 = layers.Dense(input_dim, activation='sigmoid')(decoded2)

lr = 0.01
autoencoder = models.Model(inputs=input_layer, outputs=decoded1)
optimizer_1 = Adam(learning_rate=lr)
autoencoder.compile(optimizer=optimizer_1, loss='mse')

# Pre-train autoencoder
autoencoder.fit(X_train, X_train,
                epochs=10000,
                batch_size=32,
                shuffle=True,
                validation_data=(X_test, X_test),
                verbose=1)

# Extract encoder part
encoder = models.Model(inputs=input_layer, outputs=encoded3)
X_train_encoded = encoder.predict(X_train)
X_test_encoded = encoder.predict(X_test)

# ===== 6. Classification layer =====
clf_input = layers.Input(shape=(X_train_encoded.shape[1],))
clf_hidden = layers.Dense(32, activation='relu')(clf_input)
clf_output = layers.Dense(num_classes, activation='softmax')(clf_hidden)

classifier = models.Model(inputs=clf_input, outputs=clf_output)
optimizer_2 = Adam(learning_rate=lr)
classifier.compile(optimizer=optimizer_2, loss='mse', metrics=['accuracy'])  #categorical_crossentropy

classifier.fit(X_train_encoded, y_train,
               epochs=10000,
               batch_size=32,
               validation_data=(X_test_encoded, y_test),
               verbose=1)

# ===== 7. Evaluation =====
Test_loss, Test_acc = classifier.evaluate(X_test_encoded, y_test, verbose=0)
Train_loss, Train_acc = classifier.evaluate(X_train_encoded, y_train, verbose=0)
print(f"Test Accuracy: {Test_acc:.4f}")
print(f"Train Accuracy: {Train_acc:.4f}")
