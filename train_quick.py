"""
Retrain the Deep Learning model on the NEW clean dataset.
Run: python train_quick.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

print("=" * 60)
print("🧠 TRAINING MODEL ON CLEAN DATASET")
print("=" * 60)

os.makedirs("model", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Load
df = pd.read_csv("dataset/traffic_dataset.csv")
print(f"✅ Loaded {len(df)} samples")

# Sanity check
print(f"\n📊 Sanity check:")
print(f"   phase values:  {sorted(df['phase'].unique())}")
print(f"   elapsed_green: min={df['elapsed_green'].min():.4f}  "
      f"max={df['elapsed_green'].max():.4f}  "
      f"mean={df['elapsed_green'].mean():.4f}")

if abs(df['elapsed_green'].mean()) > 10:
    print("❌ DATASET STILL CORRUPTED! Aborting.")
    exit(1)

# Features / target
feature_cols = [c for c in df.columns if c != 'action']
X = df[feature_cols].values
y = df['action'].values

print(f"\n📊 X: {X.shape}  y: {y.shape}")
print(f"   Action distribution:\n{pd.Series(y).value_counts().to_string()}")

# Split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)
print(f"\n📊 Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

# Scaler
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

print(f"\n✅ Scaler fitted")
print(f"   mean[9]={scaler.mean_[9]:.4f}  scale[9]={scaler.scale_[9]:.4f}")

# Model
model = models.Sequential([
    layers.Input(shape=(13,)),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(3, activation='softmax'),
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# Train
print("\n🚀 Training...")
history = model.fit(
    X_train_s, y_train,
    validation_data=(X_val_s, y_val),
    epochs=50,
    batch_size=64,
    callbacks=[
        callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1),
    ],
    verbose=1
)

# Evaluate
loss, acc = model.evaluate(X_test_s, y_test, verbose=0)
print(f"\n✅ Test loss: {loss:.4f}  acc: {acc:.4f}")

# Save
model.save("model/best_model.keras")
with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("✅ Saved model/best_model.keras and model/scaler.pkl")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history['accuracy'], label='train')
ax1.plot(history.history['val_accuracy'], label='val')
ax1.set_title('Accuracy'); ax1.legend(); ax1.grid(alpha=0.3)
ax2.plot(history.history['loss'], label='train')
ax2.plot(history.history['val_loss'], label='val')
ax2.set_title('Loss'); ax2.legend(); ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/training_history.png", dpi=150)
print("✅ Saved results/training_history.png")

print("\n🎉 DONE. You can now run: python run_experiment.py")