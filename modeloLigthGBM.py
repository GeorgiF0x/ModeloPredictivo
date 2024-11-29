import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
import joblib

# Cargar el dataset
csv_file = "proyectos_muestra_reducida.csv"  # Asegúrate de que este archivo esté en el mismo directorio o ajusta la ruta
data = pd.read_csv(csv_file)

# Mapear la columna "resultado" a 1 (Éxito) y 0 (Fracaso)
data["resultado"] = data["resultado"].map({"Exito": 1, "Fracaso": 0})

# Seleccionar columnas relevantes
selected_columns = [
    "certificaciones_requeridas",
    "precio_hora",
    "fortaleza_tecnologica",
    "experiencia_requerida",
    "numero_perfiles_requeridos",
    "curriculums",
    "resultado",  # Variable objetivo
]
filtered_data = data[selected_columns].copy()

# Codificar columnas categóricas
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X = filtered_data.drop(columns=["resultado"])
y = filtered_data["resultado"]

X_encoded = encoder.fit_transform(X)

# Balancear las clases con SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_encoded, y)

# Dividir los datos en conjuntos de entrenamiento y prueba (80% y 20%)
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Confirmar los tamaños de los conjuntos
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# Configurar los hiperparámetros
param_grid = {
    "num_leaves": [31, 50, 70],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [10, 15, 20],
    "n_estimators": [100, 300, 500],
    "min_child_samples": [20, 30, 50],
    "subsample": [0.7, 0.8, 1.0],
}

# Validación cruzada estratificada
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Entrenamiento del modelo con búsqueda de hiperparámetros
lgb_model = lgb.LGBMClassifier(random_state=42, class_weight="balanced", n_jobs=-1)
grid_search = GridSearchCV(
    estimator=lgb_model, param_grid=param_grid, cv=skf, scoring="roc_auc", verbose=1
)
grid_search.fit(X_train, y_train)

# Obtener el mejor modelo encontrado
best_model = grid_search.best_estimator_

# Guardar el modelo entrenado y el codificador
joblib.dump(best_model, "modelo_lgbm.pkl")
joblib.dump(encoder, "encoder.pkl")

print("Modelo exportado como 'modelo_lgbm.pkl' y codificador como 'encoder.pkl'.")

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
