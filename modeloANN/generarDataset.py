import pandas as pd
import random

# Generar datos simulados
n_samples = 1000  # Número de muestras
data = {
    "duracion": [random.randint(1, 24) for _ in range(n_samples)],
    "presupuesto": [random.uniform(10000, 5000000) for _ in range(n_samples)],
    "facturacion_anual": [random.choice([1, 2, 3]) for _ in range(n_samples)],
    "fortaleza_tecnologica": [random.choice([1, 2, 3]) for _ in range(n_samples)],
    "experiencia_requerida": [random.choice([1, 2, 3]) for _ in range(n_samples)],
    "lugar_trabajo": [random.choice([1, 2, 3]) for _ in range(n_samples)],
    "numero_perfiles_requeridos": [random.randint(1, 10) for _ in range(n_samples)],
    "precio_hora": [random.choice([1, 2, 3]) for _ in range(n_samples)],
    "volumetria": [random.choice([1, 2]) for _ in range(n_samples)],
    "tecnologias": [random.choice([1, 2, 3, 4, 5]) for _ in range(n_samples)],
    "prob_exito": [random.uniform(0, 100) for _ in range(n_samples)]
}

# Convertir a DataFrame
df = pd.DataFrame(data)

# Generar el campo "exito" basado en "prob_exito"
df["exito"] = df["prob_exito"].apply(lambda x: 1 if random.random() <= x / 100 else 0)

# Guardar en un archivo CSV
output_path = "dataset_entrenamiento.csv"
df.to_csv(output_path, index=False)

output_path
