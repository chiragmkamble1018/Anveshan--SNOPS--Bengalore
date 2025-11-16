import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------
# Configuration
# -------------------------
CSV_PATH = "AirQualityUCI.csv"
TARGET = "CO_GT"     # Correct target column
TEST_SPLIT = 0.2
RANDOM_STATE = 42

print("📌 Loading CSV file with auto-detected separator...")

# Auto-detect delimiter
with open(CSV_PATH, "r", encoding="utf-8", errors="ignore") as f:
    first_line = f.readline()
    sep = "," if first_line.count(",") > first_line.count(";") else ";"

print(f"✔ Detected separator: '{sep}'")

# Load CSV (fixed bad lines argument)
df = pd.read_csv(
    CSV_PATH,
    sep=sep,
    encoding="utf-8",
    on_bad_lines="skip"    # FIX APPLIED HERE 🚀
)

# Remove unnamed or empty columns
df = df.loc[:, ~df.columns.str.contains("Unnamed")]

print("Columns detected:", list(df.columns))

# Check if target exists
if TARGET not in df.columns:
    raise Exception(
        f"❌ Target column '{TARGET}' not found in CSV.\n"
        f"Available columns: {list(df.columns)}"
    )

# Replace invalid -200 values with NaN
df = df.replace({-200: np.nan})

# Drop rows where target is missing
df = df.dropna(subset=[TARGET])

# Select numeric columns only for ML
df = df.select_dtypes(include=[np.number])

print("✔ Using numeric columns:", list(df.columns))

# Split into features and target
X = df.drop(columns=[TARGET]).values
y = df[TARGET].values

print(f"📊 Dataset shape: X={X.shape}, y={y.shape}")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SPLIT, random_state=RANDOM_STATE
)

# -------------------------
# Model Architecture
# -------------------------
model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(32, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=["mae"]
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
loss, mae = model.evaluate(X_test, y_test)
print(f"📉 Test Loss: {loss:.4f}")
print(f"📏 Test MAE: {mae:.4f}")

# -------------------------
# Save Model
# -------------------------
model.save("air_quality_model.h5")
print("💾 Model saved as air_quality_model.h5")
