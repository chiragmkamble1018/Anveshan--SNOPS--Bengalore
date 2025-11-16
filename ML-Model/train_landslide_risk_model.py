import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------
# Configuration
# -------------------------
CSV_PATH = "regenerated_landslide_risk_dataset.csv"   # <-- your file
TARGET = None        # Will auto-detect if you don’t specify
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

# -------------------------
# Target Auto-Detection
# -------------------------
if TARGET is None:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise Exception("❌ No numeric columns found for target detection.")
    
    TARGET = numeric_cols[-1]     # choose last numeric column as target
    print(f"🎯 Auto-selected target column: {TARGET}")

# Validate target
if TARGET not in df.columns:
    raise Exception(
        f"❌ Target column '{TARGET}' not found.\nAvailable columns: {list(df.columns)}"
    )

# Replace any placeholder missing values (modify if needed)
df = df.replace({-200: np.nan})

# Drop rows with missing target
df = df.dropna(subset=[TARGET])

# Select numerics for ML
df = df.select_dtypes(include=[np.number])
print("✔ Using numeric columns:", list(df.columns))

# Split into features & target
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
model.save("landslide_risk_model.h5")
print("💾 Model saved as landslide_risk_model.h5")
