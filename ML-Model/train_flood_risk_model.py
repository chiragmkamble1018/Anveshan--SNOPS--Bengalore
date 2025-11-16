import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# -------------------------
# Configuration
# -------------------------
CSV_PATH = "flood_risk_dataset_india.csv"
TARGET = "Flood Occurred"    # target column
TEST_SPLIT = 0.2
RANDOM_STATE = 42

print("📌 Loading CSV file with auto-detected separator...")

# Auto-detect delimiter
with open(CSV_PATH, "r", encoding="utf-8", errors="ignore") as f:
    first_line = f.readline()
    sep = "," if first_line.count(",") > first_line.count(";") else ";"

print(f"✔ Detected separator: '{sep}'")

# Load CSV
df = pd.read_csv(
    CSV_PATH,
    sep=sep,
    encoding="utf-8",
    on_bad_lines="skip"
)

# Remove unnamed columns
df = df.loc[:, ~df.columns.str.contains("Unnamed")]

print("Columns detected:", list(df.columns))

# Check target exists
if TARGET not in df.columns:
    raise Exception(
        f"❌ Target column '{TARGET}' not found.\n"
        f"Available: {list(df.columns)}"
    )

# -------------------------
# Clean categorical columns
# -------------------------
categorical_cols = ["Land Cover", "Soil Type"]

print("✔ One-hot encoding categorical columns:", categorical_cols)

df = pd.get_dummies(df, columns=categorical_cols)

# -------------------------
# Select numeric columns + encoded columns
# -------------------------
df = df.replace({-200: np.nan})
df = df.dropna(subset=[TARGET])  # ensure target has no missing values
df = df.select_dtypes(include=[np.number])

print("✔ Using numeric columns:", list(df.columns))

# -------------------------
# Features & Target
# -------------------------
X = df.drop(columns=[TARGET]).values
y = df[TARGET].values  # binary classification

print(f"📊 Dataset shape: X={X.shape}, y={y.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SPLIT, random_state=RANDOM_STATE
)

# -------------------------
# Model Architecture
# -------------------------
model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")  # sigmoid for binary classification
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("🚀 Training model...")

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
]

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# -------------------------
# Evaluation
# -------------------------
loss, acc = model.evaluate(X_test, y_test)
print(f"📉 Test Loss: {loss:.4f}")
print(f"📏 Test Accuracy: {acc:.4f}")

# -------------------------
# Save Model
# -------------------------
model.save("flood_risk_model.h5")
print("💾 Model saved as flood_risk_model.h5")
