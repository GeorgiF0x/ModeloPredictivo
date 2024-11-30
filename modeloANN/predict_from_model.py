import os
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# Cargar el modelo entrenado
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'modelo_exito.h5')
preprocessor_path = os.path.join(script_dir, 'preprocessor.pkl')

# Cargar el modelo y el preprocesador
model = load_model(model_path)
preprocessor = joblib.load(preprocessor_path)
print("Modelo y preprocesador cargados correctamente.")

# Datos de entrada para predicción
new_data = pd.DataFrame({
    'duracion': [12],
    'presupuesto': [300000],
    'facturacion_anual': ['Más de 1M'],
    'fortaleza_tecnologica': ['Experto'],
    'experiencia_requerida': ['Específica'],
    'lugar_trabajo': ['Remoto'],
    'numero_perfiles_requeridos': [5],
    'precio_hora': ['Dentro del mercado'],
    'volumetria': [1]
})

# Preprocesar los datos utilizando el preprocesador cargado
new_data_processed = preprocessor.transform(new_data)

# Hacer la predicción
prediction = model.predict(new_data_processed)
print(f"Probabilidad de éxito: {prediction[0][0] * 100:.2f}%")
