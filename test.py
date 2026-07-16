# ==========================================================
# House Price Prediction using ANN (TensorFlow/Keras)
# Dataset: House Prices - Advanced Regression Techniques
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------------------------------------
# Step 1: Load Dataset
# ---------------------------------------------------------

df = pd.read_csv("train.csv")

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:", df.shape)

# ---------------------------------------------------------
# Step 2: Drop ID Column
# ---------------------------------------------------------

if "Id" in df.columns:
    df.drop("Id", axis=1, inplace=True)

# ---------------------------------------------------------
# Step 3: Handle Missing Values
# ---------------------------------------------------------

# Numeric columns
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Categorical columns
categorical_cols = df.select_dtypes(include=["object"]).columns

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing Values Remaining:")
print(df.isnull().sum().sum())

# ---------------------------------------------------------
# Step 4: Convert Categorical Data to Numeric
# ---------------------------------------------------------

df = pd.get_dummies(df, drop_first=True)

print("\nShape after One-Hot Encoding:", df.shape)

# ---------------------------------------------------------
# Step 5: Separate Features and Target
# ---------------------------------------------------------

X = df.drop("SalePrice", axis=1)


y = np.log1p(df["SalePrice"])

# ---------------------------------------------------------
# Step 6: Train-Test Split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ---------------------------------------------------------
# Step 7: Feature Scaling
# ---------------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape :", X_test.shape)

# ---------------------------------------------------------
# Step 8: Build ANN Model
# ---------------------------------------------------------

model = Sequential()

model.add(Dense(128, activation="relu", input_shape=(X_train.shape[1],)))

model.add(Dense(64, activation="relu"))

model.add(Dense(32, activation="relu"))

model.add(Dense(16, activation="relu"))

model.add(Dense(1))

# ---------------------------------------------------------
# Step 9: Compile Model
# ---------------------------------------------------------

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

# ---------------------------------------------------------
# Step 10: Early Stopping
# ---------------------------------------------------------

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

# ---------------------------------------------------------
# Step 11: Train ANN
# ---------------------------------------------------------

history = model.fit(
    X_train,
    y_train,
    validation_split=0.20,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# ---------------------------------------------------------
# Step 12: Evaluate Model
# ---------------------------------------------------------

loss, mae = model.evaluate(X_test, y_test)

print("\n==============================")
print("Test Loss :", loss)
print("Test MAE  :", mae)
print("==============================")

# ---------------------------------------------------------
# Step 13: Predict House Prices
# ---------------------------------------------------------

predictions = model.predict(X_test)

print("\nFirst 10 Predictions\n")

for i in range(10):
    print(f"Actual: {y_test.iloc[i]:,.0f}   Predicted: {predictions[i][0]:,.0f}")

# ---------------------------------------------------------
# Step 14: Plot Loss
# ---------------------------------------------------------

plt.figure(figsize=(8,5))

plt.plot(history.history["loss"], label="Training Loss")

plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Training vs Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.show()

# ---------------------------------------------------------
# Step 15: Plot MAE
# ---------------------------------------------------------

plt.figure(figsize=(8,5))

plt.plot(history.history["mae"], label="Training MAE")

plt.plot(history.history["val_mae"], label="Validation MAE")

plt.title("Training vs Validation MAE")

plt.xlabel("Epoch")

plt.ylabel("MAE")

plt.legend()

plt.grid(True)

plt.show()

# ---------------------------------------------------------
# Step 16: Save Model
# ---------------------------------------------------------

model.save("house_price_ann.keras")

print("\nModel saved successfully as house_price_ann.keras")