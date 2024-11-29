import joblib
import pandas as pd

# Cargar el modelo y el codificador
best_model = joblib.load("modelo_lgbm.pkl")
encoder = joblib.load("encoder.pkl")

# Ejemplo de datos de entrada (solo las columnas válidas para el modelo)
nuevo_proyecto = {
    "certificaciones_requeridas": True,
    "precio_hora": 50.0,
    "fortaleza_tecnologica": "Nivel Experto",
    "experiencia_requerida": "General",  # Ejemplo de valor válido
    "numero_perfiles_requeridos": 3,
    "curriculums": 10,
}

# Crear DataFrame con los datos del nuevo proyecto
nuevo_proyecto_df = pd.DataFrame([nuevo_proyecto])

# Codificar las características categóricas
nuevo_proyecto_encoded = encoder.transform(nuevo_proyecto_df)

# Predecir la probabilidad de éxito
probabilidad_exito = best_model.predict_proba(nuevo_proyecto_encoded)[:, 1][0]
print(f"Probabilidad de éxito: {probabilidad_exito:.2f}")
