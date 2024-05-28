from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Carregar o modelo treinado
model = joblib.load("house_pricing_model.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Obtém os dados do formulário
    print("Recebendo dados do formulário...")
    print("Feature 1:", request.form["feature1"])
    print("Feature 2:", request.form["feature2"])

    features = [float(request.form["feature1"]), 
                float(request.form["feature2"]),
                # Adicione as demais features aqui
               ]

    # Realiza a predição
    print("Realizando predição...")
    prediction = model.predict([features])[0]

    return render_template("result.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
