from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

@app.route('/')
def index():
    # render your HTML template here
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Mendapatkan nilai gejala dari form
    gejala = request.form.getlist('gejala')

    # Memastikan jumlah gejala antara 3 dan 5
    if len(gejala) < 3 or len(gejala) > 5:
        return "Jumlah gejala harus antara 3 dan 5."

    # Baca file Excel
    df = pd.read_csv('new_data.csv')

    # Load model
    model = joblib.load('health_model.pkl')

    # Hapus nilai kolom prognosis
    df['prognosis'] = ''

    # Isi seluruh dataframe dengan nilai 0
    df = df.applymap(lambda x: 0)

    # Isi kolom yang sesuai dengan gejala dengan 1
    for gejala_item in gejala:
        df.loc[:, gejala_item] = 1

    # Prediksi prognosis berdasarkan gejala yang dimasukkan
    new_predictions = model.predict(df.drop('prognosis', axis=1))

    # Assign the predictions to the 'prognosis' column
    df['prognosis'] = new_predictions

    # Save the updated new_data to a new CSV file
    df.to_csv('new_data.csv', index=False)

    # Membuat teks yang berisi informasi prognosis dan gejala yang dipilih
    result_text = "Prognosis: {}\\nGejala yang dipilih: {}".format(', '.join(new_predictions), ', '.join(gejala))

    # Return prognosis and selected symptoms
    return result_text

if __name__ == '__main__':
    app.run(debug=True)