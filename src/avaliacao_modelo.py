import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
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

# Dividir os dados em conjuntos de treinamento e teste
print("Dividindo os dados em conjuntos de treinamento e teste...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Carregar o scaler salvo
scaler_path = os.path.join(current_dir, '../modelos/scaler.pkl')
scaler = joblib.load(scaler_path)

# Padronizar os dados de teste
X_test_scaled = scaler.transform(X_test)

# Carregar o modelo treinado
modelo_path = os.path.join(current_dir, '../modelos/modelo_regressao_logistica.pkl')
print(f"Carregando o modelo de: {modelo_path}")
modelo = joblib.load(modelo_path)

# Fazer previsões no conjunto de teste
print("Fazendo previsões...")
y_pred = modelo.predict(X_test_scaled)

# Avaliar o modelo
print("Acurácia:", accuracy_score(y_test, y_pred))
print("Precisão:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_pred))

# Gráfico da Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title("Matriz de Confusão")
plt.savefig(os.path.join(current_dir, '../relatorios/figuras/matriz_confusao.png'))
plt.show()
