import pandas as pd
import random

# Función para generar el porcentaje de éxito basado en las características
def calculate_prob_exito(row):
    # Base del éxito
    prob = 50  # Éxito base en porcentaje

    # Ajustes por duración (proyectos más largos pueden tener menor éxito)
    if row["duracion"] <= 6:
        prob += 10
    elif row["duracion"] > 18:
        prob -= 10

    # Ajustes por presupuesto
    if row["presupuesto"] > 1000000:
        prob += 10
    elif row["presupuesto"] < 50000:
        prob -= 10

    # Ajustes por facturación anual
    if row["facturacion_anual"] == 3:  # Más de 1M
        prob += 10
    elif row["facturacion_anual"] == 1:  # Menos de 250k
        prob -= 10

    # Ajustes por fortaleza tecnológica
    if row["fortaleza_tecnologica"] == 3:  # Experto
        prob += 15
    elif row["fortaleza_tecnologica"] == 1:  # Básico
        prob -= 10

    # Ajustes por experiencia requerida
    if row["experiencia_requerida"] == 3:  # Específica
        prob += 5

    # Ajustes por lugar de trabajo
    if row["lugar_trabajo"] == 2:  # Remoto
        prob += 5

    # Ajustes por número de perfiles requeridos
    if row["numero_perfiles_requeridos"] > 5:
        prob -= 5

    # Ajustes por precio hora
    if row["precio_hora"] == 3:  # Por encima del mercado
        prob -= 5

    # Ajustes por volumetría
    if row["volumetria"] == 1:  # Sí
        prob += 5

    # Ajustes por tecnologías
    if row["tecnologias"] in [3, 4, 5]:  # Tecnologías populares
        prob += 5

    # Asegurar que esté entre 0 y 100
    return max(0, min(100, prob))

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
    "tecnologias": [random.choice([1, 2, 3, 4, 5]) for _ in range(n_samples)]
}

# Convertir a DataFrame
df = pd.DataFrame(data)

# Calcular el porcentaje de éxito para cada fila
df["prob_exito"] = df.apply(calculate_prob_exito, axis=1)

# Generar el campo "exito" basado en "prob_exito"
df["exito"] = df["prob_exito"].apply(lambda x: 1 if random.random() <= x / 100 else 0)

# Guardar en un archivo CSV
output_path = "dataset_entrenamiento.csv"
df.to_csv(output_path, index=False)

output_path
