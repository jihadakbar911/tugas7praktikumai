from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib

# =======================
# 1. Load & Preprocessing
# =======================
df = pd.read_csv("dataset/preprocessed_data_final.csv")
X = df.drop("Reflu", axis=1)
y = df["Reflu"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ==========================
# 2. Model Backpropagation
# ==========================
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# =========================
# 3. Tangani Data Imbalance
# =========================
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
cw_dict = dict(enumerate(class_weights))

# =====================
# 4. Train Model
# =====================
history = model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stop],
                    class_weight=cw_dict)

# ==========================
# 5. Evaluasi Model Lengkap
# ==========================
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss (binary crossentropy): {loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")

# Prediksi untuk metrik lainnya
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# Confusion Matrix dan Classification Report
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# Mean Squared Error
mse = mean_squared_error(y_test, y_pred_prob)
print(f"Mean Squared Error (MSE): {mse:.4f}")

# ======================
# 6. Visualisasi Training
# ======================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Grafik Loss Selama Training')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Grafik Akurasi Selama Training')
plt.legend()

plt.tight_layout()
plt.show()

# =============================
# 7. Prediksi Data Baru (3 orang)
# =============================
new_data = np.array([
    [1, 0, 0, 2, 1, 2, 2, 1, 0, 1, 3, 2, 1],
    [0, 1, 0, 4, 0, 3, 4, 0, 0, 2, 4, 1, 0],
    [0, 0, 1, 1, 2, 1, 1, 2, 1, 1, 2, 3, 2]
])

new_data_scaled = scaler.transform(new_data)
predictions = model.predict(new_data_scaled)
predicted_labels = (predictions > 0.5).astype(int)

for i, p in enumerate(predicted_labels):
    status = "BERISIKO" if p[0] == 1 else "TIDAK BERISIKO"
    print(f"Individu ke-{i+1}: {status} (Probabilitas: {predictions[i][0]:.4f})")

np.savetxt("predictions.csv", predicted_labels, delimiter=",")

# ========================
# 8. Simpan Model & Scaler
# ========================
model.save("model.h5")
joblib.dump(scaler, "scaler.pkl")
