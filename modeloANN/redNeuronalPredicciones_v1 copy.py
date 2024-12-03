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
file_path = os.path.join(script_dir, 'new_project_data_v3.csv')

# Cargar los datos
data = pd.read_csv(file_path)

# Separar características (X) y la variable objetivo (y)
X = data.drop(columns=['prob_exito'])  # Características
y = data['prob_exito'] / 100  # Normalizar el porcentaje a rango [0, 1]

# Identificar columnas categóricas y numéricas
categorical_columns = ['facturacion_anual', 'fortaleza_tecnologica', 
                       'experiencia_requerida', 'lugar_trabajo', 'precio_hora']
numerical_columns = ['duracion', 'presupuesto', 'numero_perfiles_requeridos', 'volumetria']

# Definir categorías posibles para cada columna categórica
categorical_categories = {
    'facturacion_anual': ["Menos de 250k", "250k a 1M", "Más de 1M"],
    'fortaleza_tecnologica': ["Básico", "Intermedio", "Experto"],
    'experiencia_requerida': ["Sin Experiencia", "General", "Específica"],
    'lugar_trabajo': ["Presencial", "Remoto"],
    'precio_hora': ["Por debajo del mercado", "Dentro del mercado", "Por encima del mercado"]
}

# Crear el preprocesador con manejo de categorías desconocidas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(categories=[categorical_categories[col] for col in categorical_columns],
                              handle_unknown='ignore'), categorical_columns)
    ]
)

# Preprocesar los datos
X_processed = preprocessor.fit_transform(X)

# Guardar el preprocesador
preprocessor_path = os.path.join(script_dir, 'preprocessor.pkl')
joblib.dump(preprocessor, preprocessor_path)
print(f"Preprocesador guardado en: {preprocessor_path}")

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

# Evaluar el modelo
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f'Mean Absolute Error (MAE): {mae:.2f}')

# Visualizar métricas de entrenamiento
plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.show()

plt.plot(history.history['mae'], label='MAE de entrenamiento')
plt.plot(history.history['val_mae'], label='MAE de validación')
plt.xlabel('Épocas')
plt.ylabel('MAE')
plt.legend()
plt.show()

# **Análisis visual de predicciones**
predictions = model.predict(X_test)

# Comparar valores reales vs predicciones
plt.scatter(y_test, predictions, alpha=0.6)
plt.plot([0, 1], [0, 1], '--', color='red')  # Línea ideal
plt.xlabel("Valores Reales")
plt.ylabel("Predicciones")
plt.title("Comparación: Valores Reales vs Predicciones")
plt.show()

# **Calcular otros índices**
# MSE y RMSE
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

# R² Score
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.2f}")

# Guardar el modelo entrenado
model_path = os.path.join(script_dir, 'modelo_exito.h5')
model.save(model_path)
print(f"Modelo guardado en: {model_path}")
