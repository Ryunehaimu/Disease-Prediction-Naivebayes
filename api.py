from flask import Flask, render_template, request
import pandas as pd
import modelai

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('diagnosis.blade.php')

@app.route('/predict', methods=['POST'])
def predict():
    # Mendapatkan data dari formulir web
    data = request.form.to_dict()
    
    # Membuat DataFrame dari data yang diterima
    new_data = pd.DataFrame([data])
    
    # Membaca file CSV yang berisi data asli
    original_data = pd.read_csv('new_data.csv')

    #menghapus hasil diagnosis sebelumnya
    original_data['prognosis'] = ''
    
    # Menemukan keluhan yang sesuai dalam colum dan mengisi dengan 1 , untuk keluhan yang tidak dialami akan disi 0
    for column, value in data.items():
        if column in original_data.columns:
            original_data.loc[original_data[column] == value, column] = 1
        else:
            original_data[column] = 0
    
    # Menyimpan data yang telah diperbarui ke file CSV
    original_data.to_csv('new_data.csv', index=False)
    
    # Memanggil fungsi dari modelai.py untuk memprediksi prognosis
    modelai.main()
    
    # Membaca file CSV yang telah diperbarui dengan prognosis
    updated_data = pd.read_csv('new_data.csv')
    prognosis = updated_data['prognosis']
    
    # Mengembalikan hasil prognosis ke web
    return render_template('diagnosis.blade.php', prognosis=prognosis)

if __name__ == '__main__':
    app.run(debug=True)