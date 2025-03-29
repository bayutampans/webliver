from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model yang sudah dilatih (pastikan file model.pkl sudah tersedia)
with open('liver_disease_model2.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array([
            data['age'], data['gender'], data['bmi'], data['alcohol'],
            data['smoking'], data['genetic'], data['activity'],
            data['diabetes'], data['hypertension'], data['liverTest']
        ]).reshape(1, -1)
        
        prediction = model.predict(features)
        result = "Pasien Memiliki Penyakit Liver" if prediction[0] == 1 else "Pasien Tidak Memiliki Penyakit Liver"
        
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)