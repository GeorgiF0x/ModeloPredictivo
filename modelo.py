import pandas as pd
import numpy as np
from fastai.tabular.all import *

# Crear el archivo de log limpio
open('train.log', 'w').close()

# Generar los datos
num = 10000
data = {
    'id_cliente': [np.random.randint(100, 200) for _ in range(num)],
    'duracion': [np.random.randint(1, 24) for _ in range(num)],
    'presupuesto': [round(np.random.uniform(10000, 100000), 2) for _ in range(num)],
    'solvencia_economica_empresa': [np.random.choice(['Alta', 'Media', 'Baja']) for _ in range(num)],
    'certificaciones_requeridas': [np.random.choice([True, False]) for _ in range(num)],
    'precio_hora': [round(np.random.uniform(10, 50), 2) for _ in range(num)],
    'fortaleza_tecnologica': [np.random.choice(['Alta', 'Media', 'Baja']) for _ in range(num)],
    'experiencia_requerida': [np.random.choice(['Alta', 'Media', 'Baja']) for _ in range(num)],
    'numero_perfiles_requeridos': [np.random.randint(1, 10) for _ in range(num)],
    'titulaciones_empleados': [np.random.choice(['Alta', 'Media', 'Baja']) for _ in range(num)],
    'idiomas': [np.random.choice([True, False]) for _ in range(num)],
    'plazos': [np.random.randint(1, 12) for _ in range(num)],
    'transicion_inicial': [np.random.randint(1, 6) for _ in range(num)],
    'facturacion_anual': [round(np.random.uniform(100000, 1000000), 2) for _ in range(num)],
    'resultado': [np.random.uniform(0, 1) for _ in range(num)],
}

# Crear el DataFrame
df = pd.DataFrame(data)

# Variables categóricas y numéricas
categorical_columns = [
    'solvencia_economica_empresa', 'certificaciones_requeridas',
    'fortaleza_tecnologica', 'experiencia_requerida', 'titulaciones_empleados', 'idiomas'
]
continuous_columns = [
    'duracion', 'presupuesto', 'precio_hora', 'numero_perfiles_requeridos',
    'plazos', 'transicion_inicial', 'facturacion_anual'
]

# Crear los DataLoaders con normalización y categorización
dls = TabularDataLoaders.from_df(
    df, 
    y_names="resultado", 
    cat_names=categorical_columns, 
    cont_names=continuous_columns, 
    procs=[Categorify, Normalize]
)

# Obtener los tamaños de embeddings desde los DataLoaders
emb_szs = dls.get_emb_szs()

# Crear un modelo con dropout
model = TabularModel(
    emb_szs=emb_szs,                   # Tamaño de embedding para categorías
    n_cont=len(continuous_columns),    # Número de columnas continuas
    out_sz=1,                          # Tamaño de salida (predicción de un valor continuo)
    layers=[500, 300, 100],            # Capas del modelo
    ps=[0.5, 0.5, 0.5]                 # Dropout en cada capa
)

# Crear el Learner
learn = TabularLearner(dls, model, metrics=[rmse, mae])

# Encontrar el mejor learning rate
learn.lr_find()

# Ajustar el learning rate (según lr_find())
best_lr = 1e-3  # Ajustar este valor según el gráfico de lr_find()

# Entrenar el modelo
learn.fit_one_cycle(20, lr=best_lr)

# Exportar el modelo
learn.export('c:/Users/gborisov/Desktop/modeloPredicativo/trained_model_optimized.pkl')
