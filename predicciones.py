import pandas as pd
from fastai.tabular.all import load_learner
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Cargar el modelo exportado
learn = load_learner('c:/Users/gborisov/Desktop/modeloPredicativo/trained_model.pkl')

# Generar un conjunto de datos de prueba no visto durante el entrenamiento
test_data = [
    {'id_cliente': 101, 'duracion': 36, 'presupuesto': 150000.00, 'solvencia_economica_empresa': 'Baja',
     'certificaciones_requeridas': True, 'precio_hora': 10.00, 'fortaleza_tecnologica': 'Baja',
     'experiencia_requerida': 'Alta', 'numero_perfiles_requeridos': 8, 'titulaciones_empleados': 'Media',
     'idiomas': True, 'plazos': 24, 'transicion_inicial': 5, 'facturacion_anual': 300000.00, 'resultado': 0.3},
    
    {'id_cliente': 120, 'duracion': 6, 'presupuesto': 20000.00, 'solvencia_economica_empresa': 'Media',
     'certificaciones_requeridas': False, 'precio_hora': 45.00, 'fortaleza_tecnologica': 'Alta',
     'experiencia_requerida': 'Media', 'numero_perfiles_requeridos': 4, 'titulaciones_empleados': 'Alta',
     'idiomas': False, 'plazos': 4, 'transicion_inicial': 1, 'facturacion_anual': 900000.00, 'resultado': 0.8},
    
    {'id_cliente': 199, 'duracion': 18, 'presupuesto': 75000.00, 'solvencia_economica_empresa': 'Alta',
     'certificaciones_requeridas': True, 'precio_hora': 25.00, 'fortaleza_tecnologica': 'Media',
     'experiencia_requerida': 'Media', 'numero_perfiles_requeridos': 6, 'titulaciones_empleados': 'Baja',
     'idiomas': True, 'plazos': 9, 'transicion_inicial': 3, 'facturacion_anual': 600000.00, 'resultado': 0.6},
]

# Convertir datos de prueba a DataFrame
test_df = pd.DataFrame(test_data)

# Convertir tipos de columnas según lo que espera el modelo
for col in learn.dls.cat_names:
    test_df[col] = test_df[col].astype(str)

for col in learn.dls.cont_names:
    test_df[col] = pd.to_numeric(test_df[col], errors='coerce')

# Crear un DataLoader para test
test_dl = learn.dls.test_dl(test_df)

# Realizar predicción
preds, _ = learn.get_preds(dl=test_dl)
print("Predicciones:", preds.numpy())

# Verdadero valor y predicción
true_values = test_df['resultado'].values
predicted_values = preds.numpy().flatten()

# Calcular métricas
rmse = mean_squared_error(true_values, predicted_values, squared=False)
mae = mean_absolute_error(true_values, predicted_values)
r2 = r2_score(true_values, predicted_values)

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R^2: {r2:.4f}")

# Graficar comparaciones
plt.figure(figsize=(6, 4))
plt.bar(["Real", "Predicho"], [true_values[0], predicted_values[0]], color=["blue", "orange"])
plt.title("Comparación entre Valor Real y Predicción")
plt.ylabel("Resultado")
plt.show()

# Graficar valores reales vs predichos
plt.figure(figsize=(8, 5))
plt.plot(true_values, label="Valor Real", marker='o', linestyle='-', color='blue')
plt.plot(predicted_values, label="Valor Predicho", marker='x', linestyle='--', color='orange')
plt.title("Comparación de Valores Reales vs Predichos")
plt.xlabel("Instancia")
plt.ylabel("Resultado")
plt.legend()
plt.grid(True)
plt.show()

# Graficar errores absolutos
errors = [abs(true - pred) for true, pred in zip(true_values, predicted_values)]
plt.figure(figsize=(8, 5))
plt.bar(range(len(errors)), errors, color='red', alpha=0.6)
plt.title("Errores Absolutos por Instancia")
plt.xlabel("Instancia")
plt.ylabel("Error Absoluto")
plt.grid(True)
plt.show()
