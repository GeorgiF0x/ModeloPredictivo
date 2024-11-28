import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Cargar el dataset
# Asegúrate de que el archivo "proyectos_muestra_reducida.csv" esté en el mismo directorio o ajusta la ruta
df = pd.read_csv("proyectos_sinteticos.csv")

# Mapeo de la etiqueta (target) "resultado"
df["resultado"] = df["resultado"].map({"Exito": 1, "Fracaso": 0})

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

# Definir y entrenar el modelo Random Forest
rf_model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10, class_weight="balanced")
rf_model.fit(X_train, y_train)

# Predicciones y probabilidades
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# Evaluación del modelo
accuracy_rf = accuracy_score(y_test, y_pred_rf)
classification_rep_rf = classification_report(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_prob_rf)

# Curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob_rf)

# Graficar la curva ROC
plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc_rf:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend(loc="lower right")
plt.show()

# Importancia de características
feature_names = list(numerical_cols) + list(encoder.get_feature_names_out(categorical_cols))
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

# Imprimir resultados
print("Accuracy del modelo Random Forest:", accuracy_rf)
print("\nReporte de clasificación:\n", classification_rep_rf)
print("\nMatriz de confusión:\n", conf_matrix_rf)
print(f"\nROC-AUC: {roc_auc_rf:.2f}")
print("\nImportancia de características:\n", importance_df.head(10))
