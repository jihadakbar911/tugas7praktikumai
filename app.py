from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model dan scaler sesuai urutan dan preprocessing di train_model.py
model = load_model('model.h5')
scaler = joblib.load('scaler.pkl')

# Urutan fitur harus sama persis dengan train_model.py
FEATURES = [
    'diet_Omn', 'diet_Veg', 'diet_Vegt', 'fruit', 'fat_meat',
    'homecooked', 'vegetable', 'alcohol', 'dessert', 'milk',
    'water', 'snack', 'red_meat'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil input dari form sesuai urutan fitur
    input_data = []
    for f in FEATURES:
        value = request.form.get(f, type=float)
        if value is None:
            # fallback jika ada field kosong
            value = 0.0
        input_data.append(value)

    # Ubah ke array numpy dan reshape
    input_array = np.array(input_data).reshape(1, -1)

    # Skala data sesuai scaler dari train_model.py
    scaled_input = scaler.transform(input_array)

    # Prediksi dengan model
    prediction = model.predict(scaled_input)
    prob = float(prediction[0][0])
    predicted_label = int(prob > 0.5)

    hasil = "BERISIKO ASAM LAMBUNG" if predicted_label == 1 else "TIDAK BERISIKO"
    probabilitas = f"{prob*100:.2f}%"

    return render_template('result.html', hasil=hasil, probabilitas=probabilitas)

if __name__ == '__main__':
    app.run(debug=True)
