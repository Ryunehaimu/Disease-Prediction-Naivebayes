import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import joblib

# Baca training data set
training_data = pd.read_csv('training.csv')

# Pisahkan kolom data
X = training_data.drop('prognosis', axis=1)
y = training_data['prognosis']

# Menghapus fitur konstan
selector_variance = VarianceThreshold(threshold=0)
X = selector_variance.fit_transform(X)

# Normalisasi fitur
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Bagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lakukan oversampling pada kelas minoritas
X_train_resampled, y_train_resampled = resample(X_train, y_train,
                                                replace=True,
                                                n_samples=len(X_train)*2,
                                                random_state=42)

# Lakukan feature selection
selector_kbest = SelectKBest(mutual_info_classif, k=15)
X_train_selected = selector_kbest.fit_transform(X_train_resampled, y_train_resampled)
X_test_selected = selector_kbest.transform(X_test)

# Buat model dengan Random Forest
model = RandomForestClassifier(random_state=42)

# Lakukan tuning hyperparameter dengan GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_selected, y_train_resampled)

# Pilih model terbaik
best_model = grid_search.best_estimator_
print("Best hyperparameters:", grid_search.best_params_)

# Train model dengan data latih terbaik
best_model.fit(X_train_selected, y_train_resampled)

# Model memprediksi data uji
predictions = best_model.predict(X_test_selected)

# Save model
joblib.dump(best_model, 'health_model.pkl')

# Baca data baru yang hanya berisi keluhan
new_data = pd.read_csv('new_data.csv')

# Menghapus kolom prognosis dari data new_data karena model tidak dapat membaca kolom prognosis di new_data
new_data_features = new_data.drop('prognosis', axis=1)

# Lakukan pra-pemrosesan yang sama pada new_data_features
new_data_features = selector_variance.transform(new_data_features)
new_data_features = scaler.transform(new_data_features)
new_data_selected = selector_kbest.transform(new_data_features)

# Prediksi new_data dengan model yang telah dibuat
new_predictions = best_model.predict(new_data_selected)

# Assign hasil prediksi ke kolom prognosis
new_data['prognosis'] = new_predictions
new_data.to_csv('new_data.csv', index=False)

# Menghitung akurasi model
accuracy = accuracy_score(y_test, predictions)
accuracy_percentage = accuracy * 100
print('Akurasi model:', accuracy_percentage, '%')