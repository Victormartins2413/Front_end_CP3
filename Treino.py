import pandas as pd

# Carregar os dados do arquivo CSV
df = pd.read_csv("house_pricing.csv")

# Visualizar as primeiras linhas do DataFrame
print(df.head())


# Exemplo de pré-processamento
# Lidar com valores ausentes
df.fillna(0, inplace=True)

# Codificar variáveis categóricas, se houver
# Exemplo: df = pd.get_dummies(df)

# Normalizar os dados numéricos, se necessário
# Exemplo: from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# df[['feature1', 'feature2']] = scaler.fit_transform(df[['feature1', 'feature2']])


from sklearn.model_selection import train_test_split

X = df.drop(columns=['size_house'])  # Recursos
y = df['num_bath']  # Variável de destino

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.ensemble import RandomForestRegressor

# Inicializar e treinar o modelo de regressão
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Avaliar o modelo
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Train R^2 score: {train_score}")
print(f"Test R^2 score: {test_score}")


import joblib

# Salvar o modelo treinado
joblib.dump(model, "house_pricing_model.pkl")
