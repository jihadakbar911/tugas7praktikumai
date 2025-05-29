from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib

# =======================
# 1. Load & Preprocessing
# =======================

# Muat dataset
df = pd.read_csv("dataset/preprocessed_data_final.csv")

# Pisahkan fitur dan target
X = df.drop("Reflu", axis=1)
y = df["Reflu"]

# Normalisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ==========================
# 2. Model Backpropagation
# ==========================

model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Kompilasi model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# =====================
# 3. Early Stopping
# =====================
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# =========================
# 4. Tangani Data Imbalance
# =========================
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
cw_dict = dict(enumerate(class_weights))

# =====================
# 5. Train Model
# =====================
history = model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stop],
                    class_weight=cw_dict)

# Evaluasi model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")

# ======================
# 6. Visualisasi Hasil
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

# Simpan prediksi ke CSV
np.savetxt("predictions.csv", predicted_labels, delimiter=",")

# ========================
# 8. Simpan Model & Scaler
# ========================
model.save("model.h5")
joblib.dump(scaler, "scaler.pkl")
