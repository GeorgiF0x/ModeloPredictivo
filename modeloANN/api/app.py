from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import joblib
import pandas as pd
import os

# Crear la aplicación FastAPI
app = FastAPI()

# Cargar el modelo y el preprocesador
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'modelo_exito.h5')
preprocessor_path = os.path.join(script_dir, 'preprocessor.pkl')

model = load_model(model_path)
preprocessor = joblib.load(preprocessor_path)

print("Modelo y preprocesador cargados correctamente.")

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

# Endpoint para predecir el porcentaje de éxito
@app.post("/predict/")
def predict_project(data: ProjectData):
    # Convertir la entrada en un DataFrame
    input_data = pd.DataFrame([data.dict()])
    
    # Preprocesar los datos
    try:
        input_processed = preprocessor.transform(input_data)
    except ValueError as e:
        return {"error": f"Error en el preprocesamiento: {str(e)}"}
    
    # Realizar la predicción
    prediction = model.predict(input_processed)
    
    # Convertir la predicción a porcentaje con 2 decimales y agregar el símbolo de %
    prob_exito = f"{prediction[0][0] * 100:.2f}%"
    
    return {"prob_exito": prob_exito}
