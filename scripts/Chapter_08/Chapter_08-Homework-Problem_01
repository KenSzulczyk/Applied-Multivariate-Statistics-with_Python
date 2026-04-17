# ================================
# 1. Import libraries
# ================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ================================
# 2. Load dataset
# ================================
df = pd.read_csv("Chapter_07-churn_data.csv")

# ================================
# 3. Drop missing values (as requested)
# ================================

# Convert to numeric → blanks become NaN
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Drop missing values
df = df.dropna()

# ================================
# 4. Encode target variable
# ================================
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# ================================
# 5. Encode categorical variables
# ================================
categorical_cols = [
    "Contract",
    "InternetService",
    "PaymentMethod"
]

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# ================================
# 6. Features and target
# ================================

X = df[['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']]
y = df['Churn']

# ================================
# 7. Train-test split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# 8. Scale features
# ================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================================
# 9. Build neural network
# ================================

model = keras.Sequential([
    keras.Input(shape=(X_train.shape[1],)),  # explicit input layer
    layers.Dense(16, activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# ================================
# 10. Compile model
# ================================
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ================================
# 11. Early stopping callback
# ================================
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',     # watch validation loss
    patience=5,             # stop after 5 epochs of no improvement
    restore_best_weights=True
)

# ================================
# 12. Train model
# ================================
history = model.fit(
    X_train, y_train,
    epochs=100,                  # set high, early stopping will control it
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# ================================
# 13. Evaluate model
# ================================
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy:.4f}")

# ================================
# 14. Predictions
# ================================
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# --------------------------------------------
# 15. Visualize confusion matrix
# --------------------------------------------
plt.figure()
sns.heatmap(
    pd.DataFrame(confusion_matrix(y_test, y_pred)),
    annot=True,
    fmt='g',
    cmap='Blues'
)

plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

print()
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
