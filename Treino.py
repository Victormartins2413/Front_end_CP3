from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import joblib

# Carregar conjunto de dados
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar e treinar o modelo de regressão
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Avaliar o modelo
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Train R^2 score: {train_score}")
print(f"Test R^2 score: {test_score}")

# Salvar o modelo treinado
joblib.dump(model, "gradient_boosting_model.pkl")
