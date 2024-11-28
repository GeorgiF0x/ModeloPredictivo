# Importar las librerías necesarias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Cargar el dataset generado
df = pd.read_csv("proyectos_muestra_reducida.csv")

# Mapeo de la etiqueta (target) "resultado"
df["resultado"] = df["resultado"].map({"Exito": 1, "Fracaso": 0})

# Reducir el dataset al 30% con muestreo aleatorio
df = df.sample(frac=0.3, random_state=42)

# Agregar nuevas características
df['presupuesto_duracion'] = df['presupuesto'] / df['duracion']
df['log_facturacion_anual'] = np.log1p(df['facturacion_anual'])
df['curriculums_duracion'] = df['curriculums'] * df['duracion']

# Separar features (X) y target (y)
X = df.drop(columns=["resultado", "id", "nombre_proyecto", "fecha_inicio", "fecha_fin", "informacion_adicional"])
y = df["resultado"]

# Identificar las columnas categóricas y numéricas
categorical_cols = X.select_dtypes(include=["object", "bool"]).columns
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns

# Procesar columnas categóricas con One-Hot Encoding
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_categorical = encoder.fit_transform(X[categorical_cols])

# Escalar columnas numéricas
scaler = StandardScaler()
X_numerical = scaler.fit_transform(X[numerical_cols])

# Combinar las características transformadas
X_processed = np.hstack([X_numerical, X_categorical])

# Aplicar SMOTE para balancear las clases
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_processed, y)

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Definir el modelo base para identificar características importantes
temp_model = lgb.LGBMClassifier(random_state=42)
temp_model.fit(X_train, y_train)

# Analizar la importancia de las características
feature_importances = temp_model.feature_importances_

# Mapear nombres de las características
feature_names = list(numerical_cols) + list(encoder.get_feature_names_out(categorical_cols))

# Crear un DataFrame para visualizar la importancia
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)

# Filtrar características con importancia >= 20
important_features = importance_df[importance_df["Importance"] >= 20]["Feature"].tolist()
X_train = pd.DataFrame(X_train, columns=feature_names)[important_features].values
X_test = pd.DataFrame(X_test, columns=feature_names)[important_features].values

# Definir el espacio de búsqueda para optimizar el modelo
param_grid = {
    "num_leaves": [20, 31, 40],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [5, 10, 15],
    "feature_fraction": [0.7, 0.8, 0.9],
    "bagging_fraction": [0.7, 0.8, 0.9],
    "lambda_l1": [0.0, 0.1, 0.5],
    "lambda_l2": [0.0, 0.1, 0.5],
}

# Configurar GridSearchCV
grid_search = GridSearchCV(estimator=lgb.LGBMClassifier(random_state=42), param_grid=param_grid, cv=3, scoring="accuracy", verbose=1)
grid_search.fit(X_train, y_train)

# Obtener los mejores parámetros
best_params = grid_search.best_params_

# Entrenar el modelo con los mejores parámetros
optimized_model = lgb.LGBMClassifier(**best_params, random_state=42)
optimized_model.fit(X_train, y_train)

# Predicciones con el modelo optimizado
y_pred_optimized = optimized_model.predict(X_test)
y_prob_optimized = optimized_model.predict_proba(X_test)[:, 1]

# Evaluar el modelo optimizado
accuracy_optimized = accuracy_score(y_test, y_pred_optimized)
classification_rep_optimized = classification_report(y_test, y_pred_optimized)
conf_matrix_optimized = confusion_matrix(y_test, y_pred_optimized)

# Calcular el AUC
roc_auc = roc_auc_score(y_test, y_prob_optimized)

# Curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob_optimized)

# Graficar la curva ROC
plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend(loc="lower right")
plt.show()

# Imprimir resultados
print("Accuracy del modelo optimizado:", accuracy_optimized)
print("\nReporte de clasificación:\n", classification_rep_optimized)
print("\nMatriz de confusión:\n", conf_matrix_optimized)
print(f"\nROC-AUC: {roc_auc:.2f}")

print("\nImportancia de características:\n", importance_df.head(10))
