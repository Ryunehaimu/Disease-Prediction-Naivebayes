from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)  # Enables CORS for all routes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    gejala_keys = ['gejala1', 'gejala2', 'gejala3', 'gejala4', 'gejala5']
    gejala_values = [request.form.get(key) for key in gejala_keys if request.form.get(key)]
    
    if len(gejala_values) < 3:
        return jsonify({"error": "Anda harus memasukkan minimal 3 gejala"}), 400

    # Load data and model
    df = pd.read_csv('new_data.csv')
    model = joblib.load('health_model.pkl')
    
    # Prepare the data for prediction
    df['prognosis'] = ''
    df = df.applymap(lambda x: 0)
    for gejala in gejala_values:
        if gejala in df.columns:
            df.loc[:, gejala] = 1

    # Make predictions
    new_predictions = model.predict(df.drop('prognosis', axis=1))
    df['prognosis'] = new_predictions
    df.to_csv('new_data.csv', index=False)

    # Create response data
    result_data = {
        "prognosis": new_predictions.tolist(),  # Convert numpy array to list for JSON serialization
        "selected_symptoms": gejala_values
    }

    return jsonify(result_data)

if __name__ == '__main__':
    app.run(debug=True)
