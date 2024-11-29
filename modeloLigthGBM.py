import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np

# Cargar el dataset
csv_file = "proyectos_sinteticos.csv"  # Asegúrate de que este archivo esté en el mismo directorio o ajusta la ruta
data = pd.read_csv(csv_file)

# Mapear la columna "resultado" a 1 (Éxito) y 0 (Fracaso)
data["resultado"] = data["resultado"].map({"Exito": 1, "Fracaso": 0})

# Agregar nuevas características basadas en las columnas existentes
data['ratio_cv_perfil'] = data['curriculums'] / (data['numero_perfiles_requeridos'] + 1)
data['fortaleza_certificaciones'] = ((data['fortaleza_tecnologica'] == "Nivel Experto") & 
                                      (data['certificaciones_requeridas'] == True)).astype(int)

# Seleccionar las columnas relevantes para el modelo
relevant_columns = [
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
filtered_data = data[relevant_columns].copy()

# Separar características (X) y objetivo (y)
X = filtered_data.drop(columns=["resultado"])
y = filtered_data["resultado"]

# Codificar las columnas categóricas
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_encoded = encoder.fit_transform(X)

# Balancear las clases con SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_encoded, y)

# Ajustar la división a 80% para entrenamiento y 20% para pruebas
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Confirmar los tamaños de los conjuntos
train_size = X_train.shape[0]
test_size = X_test.shape[0]

{"Train Size": train_size, "Test Size": test_size}

# Configurar los hiperparámetros para la búsqueda
param_grid = {
    "num_leaves": [31, 50, 70],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [10, 15, 20],
    "n_estimators": [100, 300, 500],
    "min_child_samples": [20, 30, 50],
    "subsample": [0.7, 0.8, 1.0]
}

# Validación cruzada estratificada
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Realizar la búsqueda de hiperparámetros con validación cruzada
lgb_model = lgb.LGBMClassifier(random_state=42, device="gpu", class_weight="balanced")
grid_search = GridSearchCV(
    estimator=lgb_model, param_grid=param_grid, cv=skf, scoring="roc_auc", verbose=1
)
grid_search.fit(X_train, y_train)

# Obtener el mejor modelo
best_model = grid_search.best_estimator_

# Evaluar el modelo con validación cruzada
cv_scores = cross_val_score(best_model, X_train, y_train, cv=skf, scoring="roc_auc")
mean_cv_score = np.mean(cv_scores)

# Realizar predicciones en el conjunto de prueba
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# Evaluar el modelo en el conjunto de prueba
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

# Mostrar la curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend(loc="lower right")
plt.show()

# Mostrar resultados
print("ROC-AUC promedio en validación cruzada:", mean_cv_score)
print("Accuracy del modelo en conjunto de prueba:", accuracy)
print("\nReporte de clasificación:\n", classification_rep)
print("\nMatriz de confusión:\n", conf_matrix)
print(f"\nROC-AUC en conjunto de prueba: {roc_auc:.2f}")
