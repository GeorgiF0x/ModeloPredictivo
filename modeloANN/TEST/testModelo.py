import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import os
# Obtener la ruta del script
script_dir = os.path.dirname(os.path.abspath(__file__))
test_dataset_path = os.path.join(script_dir, "test_dataset.csv")

# Cargar el archivo
test_data = pd.read_csv(test_dataset_path)
print(test_data.head())  # Para confirmar que el archivo se cargó correctamente

# Cargar el preprocesador de predicción
prediction_preprocessor_path = "prediction_preprocessor.pkl"
preprocessor = joblib.load(prediction_preprocessor_path)

# Separar características (X) del dataset de prueba
X_test = test_data.drop(columns=["exito", "prob_exito"])  # Quitar 'exito' y 'prob_exito'

# Preprocesar los datos de prueba
X_test_processed = preprocessor.transform(X_test)

# Ruta del modelo entrenado guardado
model_path = "modelo_exito.h5"

# Cargar el modelo
model = load_model(model_path)

# Realizar predicciones
predictions = model.predict(X_test_processed)

# Convertir las predicciones a porcentaje (0-100)
predicted_prob_exito = (predictions.flatten() * 100).round(2)

# Agregar las predicciones al DataFrame original
test_data["predicted_prob_exito"] = predicted_prob_exito

# Guardar el DataFrame actualizado con las predicciones
test_results_path = "/mnt/data/extreme_test_results.csv"
test_data.to_csv(test_results_path, index=False)
print(f"Resultados guardados en: {test_results_path}")

# Comparar predicciones con los valores reales
actual_prob_exito = test_data["prob_exito"]

# Calcular métricas
mae = mean_absolute_error(actual_prob_exito, predicted_prob_exito)
mse = mean_squared_error(actual_prob_exito, predicted_prob_exito)
rmse = np.sqrt(mse)
r2 = r2_score(actual_prob_exito, predicted_prob_exito)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.2f}")

# Comparar valores reales vs predicciones
plt.figure(figsize=(8, 8))
plt.scatter(actual_prob_exito, predicted_prob_exito, alpha=0.6, label="Datos")
plt.plot([0, 100], [0, 100], '--', color='red', label='Línea ideal')
plt.xlabel("Valores Reales (% éxito)")
plt.ylabel("Predicciones (% éxito)")
plt.title("Comparación: Valores Reales vs Predicciones")
plt.legend()
plt.show()