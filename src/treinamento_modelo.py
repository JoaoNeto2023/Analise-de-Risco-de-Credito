import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

'''@Autor: João Neto
   Data: 04/07/2024'''

# Obter o caminho absoluto do diretório atual
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construir o caminho absoluto para o arquivo dados_processados.csv
file_path = os.path.join(current_dir, '../dados/processados/dados_processados.csv')


print("Carregando dados processados...")
dados = pd.read_csv(file_path)

# Separar variáveis independentes e dependentes
X = dados.drop('Class', axis=1)  # 'Class' é a variável de destino
y = dados['Class']

# Padronizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir os dados em conjuntos de treinamento e teste
print("Dividindo os dados em conjuntos de treinamento e teste...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Treinar o modelo de regressão logística
print("Treinando o modelo de regressão logística...")
modelo = LogisticRegression(max_iter=500)
modelo.fit(X_train, y_train)

# Criar o diretório 'modelos' se não existir
modelos_dir = os.path.join(current_dir, '../modelos')
os.makedirs(modelos_dir, exist_ok=True)

# Salvar
modelo_path = os.path.join(modelos_dir, 'modelo_regressao_logistica.pkl')
joblib.dump(modelo, modelo_path)

# Salvar o scaler
scaler_path = os.path.join(current_dir, '../modelos/scaler.pkl')
joblib.dump(scaler, scaler_path)

print(f"Modelo salvo em: {modelo_path}")
print(f"Scaler salvo em: {scaler_path}")
