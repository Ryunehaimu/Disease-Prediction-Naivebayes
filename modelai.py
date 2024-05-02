import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import joblib

# baca training data set
training_data = pd.read_csv('training.csv')

# pisahkan colum data
X_train = training_data.drop('prognosis', axis=1)
y_train = training_data['prognosis']

# baca testing data set
test_data = pd.read_csv('testing.csv')

# pisahkan colum data
X_test = test_data.drop('prognosis', axis=1)
y_test = test_data['prognosis']

# buat model dengan Gaussian Naive Bayes
model = GaussianNB()

# Train model
model.fit(X_train, y_train)

# model mempredik data test
predictions = model.predict(X_test)

# Save model
joblib.dump(model, 'health_model.pkl')

# baca data baru yang hanya berisi keluhan
new_data = pd.read_csv('new_data.csv')

# menghapus colum prognosis dari data new_data karena model tidak dapat membaca colum prognosis di new_data
new_data_features = new_data.drop('prognosis', axis=1)

# predik new_data dengan model yang telah dibikin tadi
new_predictions = model.predict(new_data_features)

# assign hasil diatas ke column prognosis
new_data['prognosis'] = new_predictions
new_data.to_csv('new_data.csv', index=False)

# menghitung akurasi model
accuracy = accuracy_score(y_test, predictions)
accuracy_percentage = accuracy * 100

print('Akurasi model:', accuracy_percentage, '%')