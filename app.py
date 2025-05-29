from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model dan scaler
model = load_model('model.h5')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil input dari form
    input_data = [float(request.form[f]) for f in [
        'diet_Omn', 'diet_Veg', 'diet_Vegt', 'fruit', 'fat_meat',
        'homecooked', 'vegetable', 'alcohol', 'dessert', 'milk',
        'water', 'snack', 'red_meat'
    ]]

    # Normalisasi dan prediksi
    scaled_input = scaler.transform([input_data])
    prediction = model.predict(scaled_input)
    predicted_label = int(prediction[0][0] > 0.5)
    
    hasil = "BERISIKO ASAM LAMBUNG" if predicted_label == 1 else "TIDAK BERISIKO"
    probabilitas = f"{prediction[0][0]*100:.2f}%"

    return render_template('result.html', hasil=hasil, probabilitas=probabilitas)

if __name__ == '__main__':
    app.run(debug=True)
