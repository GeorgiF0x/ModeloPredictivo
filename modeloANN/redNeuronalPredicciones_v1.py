import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os
import matplotlib.pyplot as plt
import numpy as np
import joblib

# Verificar si TensorFlow detecta la GPU
print("Dispositivos detectados:", tf.config.list_physical_devices())

# Obtener la ruta absoluta del script
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'dataset_v9.csv')

# Cargar los datos
data = pd.read_csv(file_path)

# Separar características (X) y la variable objetivo (y)
X = data.drop(columns=['prob_exito'])  # Características, incluyendo 'exito'
y = data['prob_exito'] / 100  # Normalizar el porcentaje a rango [0, 1]

# Identificar columnas categóricas y numéricas
categorical_columns = ['facturacion_anual', 'fortaleza_tecnologica', 
                       'experiencia_requerida', 'lugar_trabajo', 'precio_hora']
numerical_columns = ['duracion', 'presupuesto', 'numero_perfiles_requeridos', 'volumetria', 'exito']  # Incluye 'exito'

# Definir categorías posibles para cada columna categórica
categorical_categories = {
    'facturacion_anual': ["Menos de 250k", "250k a 1M", "Más de 1M"],
    'fortaleza_tecnologica': ["Básico", "Intermedio", "Experto"],
    'experiencia_requerida': ["Sin Experiencia", "General", "Específica"],
    'lugar_trabajo': ["Presencial", "Remoto", "Híbrido"],  # Añadido 'Híbrido'
    'precio_hora': ["Por debajo del mercado", "Dentro del mercado", "Por encima del mercado"]
}

# Preprocesador para entrenamiento (incluye 'exito')
train_preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),  # Incluye 'exito'
        ('cat', OneHotEncoder(categories=[categorical_categories[col] for col in categorical_columns],
                              handle_unknown='ignore'), categorical_columns)
    ]
)

# Preprocesador para predicción (excluye 'exito')
prediction_preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), [col for col in numerical_columns if col != 'exito']),  # Excluye 'exito'
        ('cat', OneHotEncoder(categories=[categorical_categories[col] for col in categorical_columns],
                              handle_unknown='ignore'), categorical_columns)
    ]
)

# Preprocesar los datos para entrenamiento
X_processed = train_preprocessor.fit_transform(X)

# Guardar ambos preprocesadores
train_preprocessor_path = os.path.join(script_dir, 'train_preprocessor.pkl')
prediction_preprocessor_path = os.path.join(script_dir, 'prediction_preprocessor.pkl')
joblib.dump(train_preprocessor, train_preprocessor_path)
joblib.dump(prediction_preprocessor, prediction_preprocessor_path)
print(f"Preprocesadores guardados en: {train_preprocessor_path} y {prediction_preprocessor_path}")

# Dividir datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Crear el modelo
model = Sequential()

# Capa de entrada + Capa oculta 1
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))

# Capa oculta 2
model.add(Dense(32, activation='relu'))

# Capa de salida
model.add(Dense(1, activation='sigmoid'))  # Salida normalizada entre 0 y 1

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Forzar el uso de la GPU (si está disponible)
try:
    with tf.device('/GPU:0'):
        # Entrenar el modelo
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=32,
            verbose=1
        )
except RuntimeError as e:
    print("Error al configurar la GPU:", e)
    print("Entrenando en CPU en su lugar...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        verbose=1
    )

# Guardar el modelo entrenado
model_path = os.path.join(script_dir, 'modelo_exito.h5')
model.save(model_path)
print(f"Modelo guardado en: {model_path}")

# Función de preprocesamiento para predicción
def preprocess_for_prediction(input_data, preprocessor):
    # Si 'exito' no está en los datos, agregarlo con un valor predeterminado
    if 'exito' not in input_data.columns:
        input_data['exito'] = 0  # Valor predeterminado
    return preprocessor.transform(input_data)

# Función de predicción
def predict_project(input_json, model, preprocessor):
    input_data = pd.DataFrame([input_json])
    input_processed = preprocess_for_prediction(input_data, preprocessor)
    prediction = model.predict(input_processed)
    return prediction
