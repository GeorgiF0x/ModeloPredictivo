import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np

# Cargar el archivo CSV
csv_file = "proyectos_sinteticos.csv"  # Asegúrate de que este archivo esté en el mismo directorio o ajusta la ruta
data = pd.read_csv(csv_file)

# Mapear la columna "resultado" a 1 (Éxito) y 0 (Fracaso)
data["resultado"] = data["resultado"].map({"Exito": 1, "Fracaso": 0})

# Agregar nuevas características basadas en el conocimiento del dominio
data['ratio_cv_perfil'] = data['curriculums'] / (data['numero_perfiles_requeridos'] + 1)
data['fortaleza_certificaciones'] = ((data['fortaleza_tecnologica'] == "Nivel Experto") & 
                                      (data['certificaciones_requeridas'] == True)).astype(int)

# Seleccionar las columnas relevantes
columnas_relevantes = [
    "id_cliente",
    "certificaciones_requeridas",
    "precio_hora",
    "fortaleza_tecnologica",
    "experiencia_requerida",
    "numero_perfiles_requeridos",
    "curriculums",
    "titulaciones_empleados",
    "ratio_cv_perfil",
    "fortaleza_certificaciones",
    "resultado"  # Variable objetivo
]
data_filtrada = data[columnas_relevantes].copy()

# Separar características (X) y variable objetivo (y)
X = data_filtrada.drop(columns=["resultado"])
y = data_filtrada["resultado"]

# Codificar las variables categóricas
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_codificado = encoder.fit_transform(X)

# Balancear las clases usando SMOTE
smote = SMOTE(random_state=42)
X_balanceado, y_balanceado = smote.fit_resample(X_codificado, y)

# Dividir los datos en 80% para entrenamiento y 20% para prueba
X_train, X_test, y_train, y_test = train_test_split(X_balanceado, y_balanceado, test_size=0.2, random_state=42)

# Estandarizar las características
escalador = StandardScaler()
X_train = escalador.fit_transform(X_train)
X_test = escalador.transform(X_test)

# Definir y entrenar el modelo de regresión logística
modelo_reg_log = LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000)
modelo_reg_log.fit(X_train, y_train)

# Validación cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
puntajes_cv = cross_val_score(modelo_reg_log, X_train, y_train, cv=cv, scoring="roc_auc")
puntaje_promedio_cv = np.mean(puntajes_cv)

# Hacer predicciones en el conjunto de prueba
y_pred = modelo_reg_log.predict(X_test)
y_prob = modelo_reg_log.predict_proba(X_test)[:, 1]

# Evaluar el modelo
exactitud = accuracy_score(y_test, y_pred)
reporte_clasificacion = classification_report(y_test, y_pred)
matriz_confusion = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

# Graficar la curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"Curva ROC (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("Tasa de Falsos Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.title("Curva ROC (Receiver Operating Characteristic)")
plt.legend(loc="lower right")
plt.show()

# Mostrar resultados
print("ROC-AUC promedio (validación cruzada):", puntaje_promedio_cv)
print("Exactitud en el conjunto de prueba:", exactitud)
print("\nReporte de clasificación:\n", reporte_clasificacion)
print("\nMatriz de confusión:\n", matriz_confusion)
print(f"\nROC-AUC en el conjunto de prueba: {roc_auc:.2f}")
