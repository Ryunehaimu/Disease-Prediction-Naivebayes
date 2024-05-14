import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.utils import resample
import joblib

# Baca training data set
training_data = pd.read_csv('training.csv')

# Pisahkan kolom data
X = training_data.drop('prognosis', axis=1)
y = training_data['prognosis']

# Menghapus fitur konstan
selector_variance = VarianceThreshold(threshold=0)
X = selector_variance.fit_transform(X)

# Bagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lakukan oversampling pada kelas minoritas
X_train_resampled, y_train_resampled = resample(X_train, y_train,
                                                replace=True,
                                                n_samples=len(X_train)*2,
                                                random_state=42)

# Lakukan feature selection
selector_kbest = SelectKBest(mutual_info_classif, k=10)
X_train_selected = selector_kbest.fit_transform(X_train_resampled, y_train_resampled)
X_test_selected = selector_kbest.transform(X_test)

# Buat model dengan Gaussian Naive Bayes dan regularisasi
model = GaussianNB(priors=None)

# Lakukan cross-validation untuk memilih model terbaik
scores = cross_val_score(model, X_train_selected, y_train_resampled, cv=5)
print("Cross-validation scores:", scores)
print("Mean cross-validation score:", scores.mean())

# Train model dengan data latih terbaik
model.fit(X_train_selected, y_train_resampled)

# Model memprediksi data uji
predictions = model.predict(X_test_selected)

# Save model
joblib.dump(model, 'health_model.pkl')

# Baca data baru yang hanya berisi keluhan
new_data = pd.read_csv('new_data.csv')

# Menghapus kolom prognosis dari data new_data karena model tidak dapat membaca kolom prognosis di new_data
new_data_features = new_data.drop('prognosis', axis=1)

# Lakukan pra-pemrosesan yang sama pada new_data_features
new_data_features = selector_variance.transform(new_data_features)
new_data_selected = selector_kbest.transform(new_data_features)

# Prediksi new_data dengan model yang telah dibuat
new_predictions = model.predict(new_data_selected)

# Assign hasil prediksi ke kolom prognosis
new_data['prognosis'] = new_predictions
new_data.to_csv('new_data.csv', index=False)

# Menghitung akurasi model
accuracy = accuracy_score(y_test, predictions)
accuracy_percentage = accuracy * 100
print('Akurasi model:', accuracy_percentage, '%')