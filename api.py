from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Mendapatkan data JSON dari permintaan POST
    gejala_values = data.get('gejala', [])  # Mendapatkan gejala dari data JSON
    
    if len(gejala_values) < 3:
        return jsonify({"error": "Anda harus memasukkan minimal 3 gejala"}), 400

    # Load data and model
    df = pd.read_csv('new_data.csv')
    model = joblib.load('health_model.pkl')
    
    # Persiapkan data untuk prediksi
    df['prognosis'] = ''
    df = df.applymap(lambda x: 0)
    for gejala in gejala_values:
        if gejala in df.columns:
            df.loc[:, gejala] = 1

    # Lakukan prediksi
    new_predictions = model.predict(df.drop('prognosis', axis=1))
    df['prognosis'] = new_predictions
    df.to_csv('new_data.csv', index=False)

    # Buat data respons
    result_data = {
        "prognosis": new_predictions.tolist(),  # Konversi array numpy ke list untuk serialisasi JSON
        "selected_symptoms": gejala_values
    }

    return jsonify(result_data)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
