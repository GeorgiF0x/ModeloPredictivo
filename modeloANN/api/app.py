from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Literal
from tensorflow.keras.models import load_model
import joblib
import pandas as pd
import os
from fastapi.middleware.cors import CORSMiddleware

# Crear la aplicación FastAPI
app = FastAPI()

# Configurar CORS (puedes limitar las IPs permitidas para mayor seguridad)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todas las IPs
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos HTTP (GET, POST, etc.)
    allow_headers=["*"],  # Permitir todos los encabezados
)

# Obtener la ruta absoluta del script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Rutas de los modelos guardados
model_path = os.path.join(script_dir, 'modelo_exito.h5')
preprocessor_path = os.path.join(script_dir, 'prediction_preprocessor.pkl')

# Cargar el modelo y el preprocesador
try:
    model = load_model(model_path)
    preprocessor = joblib.load(preprocessor_path)
    print("Modelo y preprocesador cargados correctamente.")
except Exception as e:
    print(f"Error al cargar el modelo o preprocesador: {e}")
    model, preprocessor = None, None

# Clase para validar los datos de entrada
class ProjectData(BaseModel):
    duracion: int
    presupuesto: float
    facturacion_anual: Literal["1", "2", "3"]
    fortaleza_tecnologica: Literal["1", "2", "3"]
    experiencia_requerida: Literal["1", "2", "3"]
    lugar_trabajo: Literal["1", "2", "3"]
    numero_perfiles_requeridos: int
    precio_hora: Literal["1", "2", "3"]
    volumetria: Literal["1", "2"]
    tecnologias: List[Literal["1", "2", "3", "4", "5"]]  # Lista de tecnologías permitidas

# Función para preprocesar los datos para predicción
def preprocess_for_prediction(input_data: pd.DataFrame, preprocessor):
    # Verificar si todas las columnas necesarias están presentes
    required_columns = [
        'duracion', 'presupuesto', 'facturacion_anual', 
        'fortaleza_tecnologica', 'experiencia_requerida', 
        'lugar_trabajo', 'numero_perfiles_requeridos', 
        'precio_hora', 'volumetria', 'tecnologias'
    ]
    missing_columns = [col for col in required_columns if col not in input_data.columns]

    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas: {missing_columns}")

    # Convertir tecnologías de lista a una cadena separada por comas (si el modelo lo necesita)
    input_data['tecnologias'] = input_data['tecnologias'].apply(lambda x: ','.join(x) if isinstance(x, list) else x)

    # Asegurarse de que las columnas tengan el tipo correcto
    input_data = input_data.astype({
        'duracion': int,
        'presupuesto': float,
        'facturacion_anual': str,
        'fortaleza_tecnologica': str,
        'experiencia_requerida': str,
        'lugar_trabajo': str,
        'numero_perfiles_requeridos': int,
        'precio_hora': str,
        'volumetria': int,
        'tecnologias': str
    })

    try:
        return preprocessor.transform(input_data)
    except Exception as e:
        raise ValueError(f"Error durante el preprocesamiento: {str(e)}")

# Endpoint para predecir el porcentaje de éxito
@app.post("/predict")
def predict_project(data: ProjectData):
    # Verificar si el modelo y el preprocesador están cargados
    if not model or not preprocessor:
        raise HTTPException(status_code=500, detail="Modelo o preprocesador no cargado correctamente.")

    # Convertir el JSON recibido en un DataFrame
    input_data = pd.DataFrame([data.dict()])
    
    # Preprocesar los datos
    try:
        input_processed = preprocess_for_prediction(input_data, preprocessor)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Error en el preprocesamiento: {str(e)}")
    
    # Realizar la predicción
    try:
        prediction = model.predict(input_processed)
        # Convertir la predicción a porcentaje (0-100) con 2 decimales
        prob_exito = f"{prediction[0][0] * 100:.2f}"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")
    
    return {
        "prob_exito": prob_exito
    }
