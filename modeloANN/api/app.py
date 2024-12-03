from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import joblib
import pandas as pd
import os
from fastapi.middleware.cors import CORSMiddleware

# Crear la aplicación FastAPI
app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todas las IPs. Cambia "*" por una lista específica para mayor seguridad.
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos HTTP (GET, POST, etc.).
    allow_headers=["*"],  # Permitir todos los encabezados.
)

# Cargar el modelo y el preprocesador para predicciones
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'modelo_exito_v2.h5')
preprocessor_path = os.path.join(script_dir, 'prediction_preprocessor.pkl')

try:
    model = load_model(model_path)
    preprocessor = joblib.load(preprocessor_path)
    print("Modelo y preprocesador para predicciones cargados correctamente.")
except Exception as e:
    print(f"Error al cargar el modelo o preprocesador: {e}")
    model, preprocessor = None, None

# Clase para validar las entradas de la API
class ProjectData(BaseModel):
    duracion: int
    presupuesto: float
    facturacion_anual: str
    fortaleza_tecnologica: str
    experiencia_requerida: str
    lugar_trabajo: str
    numero_perfiles_requeridos: int
    precio_hora: str
    volumetria: int

# Preprocesamiento de datos para predicción
def preprocess_for_prediction(input_data, preprocessor):
    # Si 'exito' no está presente, agregarla con un valor predeterminado
    if 'exito' not in input_data.columns:
        input_data['exito'] = 0  # Valor predeterminado para predicciones
    return preprocessor.transform(input_data)

# Endpoint para predecir el porcentaje de éxito
@app.post("/predict")
@app.post("/predict/")
def predict_project(data: ProjectData):
    # Verificar si el modelo y el preprocesador están cargados
    if not model or not preprocessor:
        raise HTTPException(status_code=500, detail="Modelo o preprocesador no cargado correctamente.")

    # Convertir la entrada en un DataFrame
    input_data = pd.DataFrame([data.dict()])
    
    # Preprocesar los datos
    try:
        input_processed = preprocess_for_prediction(input_data, preprocessor)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Error en el preprocesamiento: {str(e)}")
    
    # Realizar la predicción
    try:
        prediction = model.predict(input_processed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")
    
    # Convertir la predicción a porcentaje con 2 decimales y agregar el símbolo de %
    prob_exito = f"{prediction[0][0] * 100:.2f}%"
    
    return {"prob_exito": prob_exito}
