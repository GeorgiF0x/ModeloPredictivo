from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
import numpy as np

app = Flask(__name__)

# Cargar el modelo y el encoder
base_path = os.path.dirname(os.path.abspath(__file__))  # Ruta absoluta del directorio actual
model_path = os.path.join(base_path, "modelo_lgbm.pkl")
encoder_path = os.path.join(base_path, "encoder.pkl")

try:
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    print("Modelo y encoder cargados correctamente.")
    print("Categorías reconocidas por el encoder:", encoder.categories_)
except Exception as e:
    print(f"Error al cargar el modelo o el encoder: {e}")
    raise e  # Detener la ejecución si no se pueden cargar

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Recibir los datos enviados en formato JSON
        data = request.get_json()
        print("Datos recibidos:", data)

        # Campos requeridos por el modelo
        required_fields = [
            "certificaciones_requeridas",
            "precio_hora",
            "fortaleza_tecnologica",
            "experiencia_requerida",
            "numero_perfiles_requeridos",
            "curriculums"
        ]

        # Validar que todos los campos requeridos estén presentes
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Faltan los campos requeridos: {', '.join(missing_fields)}"}), 400

        # Crear un DataFrame con los datos de entrada
        input_data = pd.DataFrame([data])

        # Formatear los datos numéricos a 2 decimales
        formatted_data = input_data.round(2)
        print("Datos formateados:", formatted_data.to_dict(orient="records"))

        # Validar valores categóricos en las columnas
        for i, column in enumerate(encoder.feature_names_in_):
            if column in formatted_data.columns:
                valid_values = encoder.categories_[i]
                if formatted_data[column].iloc[0] not in valid_values:
                    return jsonify({
                        "error": f"Valor no reconocido en '{column}'.",
                        "categorias_reconocidas": valid_values.tolist()
                    }), 400

        # Codificar las características categóricas
        try:
            input_encoded = encoder.transform(formatted_data)
        except Exception as e:
            return jsonify({"error": f"Error al codificar los datos: {e}"}), 400

        # Realizar la predicción
        try:
            probability = model.predict_proba(input_encoded)[:, 1][0]
        except Exception as e:
            return jsonify({"error": f"Error al predecir: {e}"}), 400

        # Convertir la probabilidad a porcentaje con 2 decimales
        probability_percentage = round(probability * 100, 2)
        formatted_percentage = f"{probability_percentage:.2f}".replace(".", ",") + "%"

        # Devolver la probabilidad de éxito en ambos formatos
        return jsonify({
            "probabilidad_decimal": round(probability, 2),
            "probabilidad_porcentaje": formatted_percentage
        })

    except Exception as e:
        print("Error en el servidor:", str(e))
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
