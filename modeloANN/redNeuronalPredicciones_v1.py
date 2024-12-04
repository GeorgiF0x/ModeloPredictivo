import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Verificar si TensorFlow detecta la GPU
print("Dispositivos detectados:", tf.config.list_physical_devices())

# Obtener la ruta absoluta del script
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'dataset_entrenamiento.csv')

# Cargar los datos
data = pd.read_csv(file_path)

# Separar características (X) y la variable objetivo (y)
X = data.drop(columns=['prob_exito', 'exito'], errors='ignore')  # Características
y = data['prob_exito'] / 100  # Normalizar el porcentaje a rango [0, 1]

# Identificar columnas categóricas y numéricas
categorical_columns = [
    'facturacion_anual', 'fortaleza_tecnologica',
    'experiencia_requerida', 'lugar_trabajo',
    'precio_hora', 'tecnologias'  # Incluye IDs de tecnologías
]
numerical_columns = [
    'duracion', 'presupuesto', 'numero_perfiles_requeridos', 'volumetria'
]

# Preprocesador para entrenamiento
train_preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ]
)

# Ajustar el preprocesador para entrenamiento
X_processed = train_preprocessor.fit_transform(X)

# Guardar el preprocesador de entrenamiento
train_preprocessor_path = os.path.join(script_dir, 'train_preprocessor.pkl')
joblib.dump(train_preprocessor, train_preprocessor_path)
print(f"Preprocesador de entrenamiento guardado en: {train_preprocessor_path}")

# Preprocesador para predicción (sin columna 'exito')
prediction_preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ]
)

# Ajustar el preprocesador para predicción
prediction_preprocessor.fit(X)

# Guardar el preprocesador de predicción
prediction_preprocessor_path = os.path.join(script_dir, 'prediction_preprocessor.pkl')
joblib.dump(prediction_preprocessor, prediction_preprocessor_path)
print(f"Preprocesador de predicción guardado en: {prediction_preprocessor_path}")

# Dividir datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Crear el modelo
model = Sequential()

model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))  # Dropout del 30%
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Capa de salida (normalizada a [0, 1])

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Early Stopping para evitar sobreajuste
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Forzar el uso de la GPU (si está disponible)
try:
    with tf.device('/GPU:0'):
        # Entrenar el modelo
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )
except RuntimeError as e:
    print("Error al configurar la GPU:", e)
    print("Entrenando en CPU en su lugar...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

# Guardar el modelo entrenado
model_path = os.path.join(script_dir, 'modelo_exito.h5')
model.save(model_path)
print(f"Modelo guardado en: {model_path}")

# Evaluar el modelo en el conjunto de prueba
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE (Error Absoluto Medio): {mae:.4f}")
print(f"MSE (Error Cuadrático Medio): {mse:.4f}")
print(f"R2 Score: {r2:.4f}")

# Graficar la pérdida y MAE durante el entrenamiento
plt.figure(figsize=(12, 6))

# Pérdida
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida Durante el Entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida (Loss)')
plt.legend()

# MAE
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Entrenamiento')
plt.plot(history.history['val_mae'], label='Validación')
plt.title('Error Absoluto Medio (MAE) Durante el Entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'training_metrics.png'))
plt.show()

# Graficar predicciones vs valores reales
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.title('Predicciones vs Valores Reales')
plt.xlabel('Valores Reales (y_test)')
plt.ylabel('Predicciones (y_pred)')
plt.savefig(os.path.join(script_dir, 'predictions_vs_actuals.png'))
plt.show()
