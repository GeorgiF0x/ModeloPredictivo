import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os
import joblib

# Verificar si TensorFlow detecta la GPU
print("Dispositivos detectados:", tf.config.list_physical_devices())

# Obtener la ruta absoluta del script
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'datos_entrenamiento.csv')

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

# Ajustar el preprocesador para entrenamiento
X_processed = train_preprocessor.fit_transform(X)

# Guardar el preprocesador de entrenamiento
train_preprocessor_path = os.path.join(script_dir, 'train_preprocessor.pkl')
joblib.dump(train_preprocessor, train_preprocessor_path)
print(f"Preprocesador de entrenamiento guardado en: {train_preprocessor_path}")

# Preprocesador para predicción (incluye 'exito' con valor predeterminado si no está presente)
prediction_preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),  # Incluye 'exito'
        ('cat', OneHotEncoder(categories=[categorical_categories[col] for col in categorical_columns],
                              handle_unknown='ignore'), categorical_columns)
    ]
)

# Ajustar el preprocesador para predicción
X_with_exito = X.copy()  # Asegurarse de que incluye la columna 'exito'
prediction_preprocessor.fit(X_with_exito)

# Guardar el preprocesador de predicción
prediction_preprocessor_path = os.path.join(script_dir, 'prediction_preprocessor.pkl')
joblib.dump(prediction_preprocessor, prediction_preprocessor_path)
print(f"Preprocesador de predicción guardado en: {prediction_preprocessor_path}")

# Dividir datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Crear el modelo
model = Sequential()

# Dar mayor peso al presupuesto agregando más capas y neuronas
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))  # Más neuronas en la entrada
model.add(Dense(64, activation='relu'))  # Segunda capa oculta
model.add(Dense(32, activation='relu'))  # Tercera capa oculta
model.add(Dense(1, activation='sigmoid'))  # Capa de salida (normalizada a [0, 1])

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
