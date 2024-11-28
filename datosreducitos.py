from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Implementar nuevas características
df['presupuesto_duracion'] = df['presupuesto'] / df['duracion']
df['log_facturacion_anual'] = np.log1p(df['facturacion_anual'])
df['curriculums_duracion'] = df['curriculums'] * df['duracion']

# Seleccionar características relevantes basadas en importancia
selected_features = [
    "curriculums", "facturacion_anual", "duracion", "transicion_inicial",
    "precio_hora", "presupuesto", "numero_perfiles_requeridos",
    "presupuesto_duracion", "log_facturacion_anual", "curriculums_duracion"
]
X = df[selected_features]
y = df["resultado"]

# Escalar características numéricas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar SMOTE para balancear las clases
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_scaled, y)

# Crear un nuevo dataset reducido
reduced_df = pd.DataFrame(X_balanced, columns=selected_features)
reduced_df["resultado"] = y_balanced

# Guardar el CSV reducido
reduced_csv_path = "proyectos_reducidos.csv"
reduced_df.to_csv(reduced_csv_path, index=False)

print(f"Dataset reducido guardado en: {reduced_csv_path}")
