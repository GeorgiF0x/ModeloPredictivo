import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
import os
import matplotlib.pyplot as plt
import numpy as np
import joblib

# Verificar si TensorFlow detecta dispositivos disponibles
print("Dispositivos detectados:", tf.config.list_physical_devices())

# Obtener la ruta absoluta del script
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'dataset_v8.csv')

# Cargar los datos
data = pd.read_csv(file_path)

# Separar características (X) y la variable objetivo (y)
X = data.drop(columns=['prob_exito'])  # Características
y = data['prob_exito'] / 100  # Normalizar el porcentaje a rango [0, 1]

# Identificar columnas categóricas y numéricas
categorical_columns = ['facturacion_anual', 'fortaleza_tecnologica', 
                       'experiencia_requerida', 'lugar_trabajo', 'precio_hora']
numerical_columns = ['duracion', 'presupuesto', 'numero_perfiles_requeridos', 'volumetria', 'exito']

# Clase personalizada para crear características derivadas dinámicamente
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['presupuesto_por_duracion'] = X['presupuesto'] / (X['duracion'] + 1e-9)
        X['perfiles_por_presupuesto'] = X['numero_perfiles_requeridos'] / (X['presupuesto'] + 1e-9)
        return X

# Crear el pipeline de preprocesamiento
preprocessor = Pipeline(steps=[
    ('feature_engineering', FeatureEngineer()),
    ('transformer', ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_columns + ['presupuesto_por_duracion', 'perfiles_por_presupuesto']),
            ('cat', OneHotEncoder(categories=[
                ["Menos de 250k", "250k a 1M", "Más de 1M"], 
                ["Básico", "Intermedio", "Experto"], 
                ["Sin Experiencia", "General", "Específica"], 
                ["Presencial", "Remoto", "Híbrido"],
                ["Por debajo del mercado", "Dentro del mercado", "Por encima del mercado"]
            ], handle_unknown='ignore'), categorical_columns)
        ]
    ))
])

# Preprocesar los datos
X_processed = preprocessor.fit_transform(X)

# Dividir datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Guardar el preprocesador
preprocessor_path = os.path.join(script_dir, 'preprocessor_v4.pkl')
joblib.dump(preprocessor, preprocessor_path)
print(f"Preprocesador guardado en: {preprocessor_path}")

# Crear el modelo con arquitectura simplificada
model = Sequential()

# Capa de entrada + Capa oculta 1
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.005)))
model.add(Dropout(0.2))  # Apaga el 20% de las neuronas en esta capa

# Capa oculta 2
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.005)))
model.add(Dropout(0.2))  # Apaga el 20% de las neuronas en esta capa

# Capa de salida
model.add(Dense(1, activation='sigmoid'))  # Salida para regresión normalizada

# Compilar el modelo con tasa de aprendizaje ajustada y función de pérdida MSE
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

# Entrenar el modelo
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
plt.ylabel('Pérdida (MSE)')
plt.legend()
plt.show()

plt.plot(history.history['mae'], label='MAE de entrenamiento')
plt.plot(history.history['val_mae'], label='MAE de validación')
plt.xlabel('Épocas')
plt.ylabel('MAE')
plt.legend()
plt.show()

# **Análisis visual de predicciones**
predictions = model.predict(X_test).flatten()

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
model_path = os.path.join(script_dir, 'modelo_exito_v4_regresion.keras')
model.save(model_path)
print(f"Modelo guardado en: {model_path}")
