from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)

# Konfigurasi CORS yang lebih ketat untuk produksi
CORS(app, resources={
    r"/predict": {
        "origins": ["*"],  # allow dari semua url/semua orang bisa hit
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Load model yang sudah dilatih
try:
    with open('liver_disease_model2.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        # Handling CORS request
        response = jsonify({'status': 'preflight'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
    if model is None:
        return jsonify({'error': 'Tidak ada model'}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Tidak ada data request'}), 400
            
        # Validasi input
        required_fields = ['age', 'gender', 'bmi', 'alcohol', 'smoking', 
                         'genetic', 'activity', 'diabetes', 'hypertension', 'liverTest']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Input tidak ditemukan: {field}'}), 400
        
        features = np.array([
            data['age'], data['gender'], data['bmi'], data['alcohol'],
            data['smoking'], data['genetic'], data['activity'],
            data['diabetes'], data['hypertension'], data['liverTest']
        ]).reshape(1, -1)
        
        prediction = model.predict(features)
        result = "Pasien Memiliki Penyakit Liver" if prediction[0] == 1 else "Pasien Tidak Memiliki Penyakit Liver"
        
        response = jsonify({'prediction': result})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)