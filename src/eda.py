import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

'''@Autor: João Neto
   Data: 04/07/2024'''

# Obter o caminho absoluto do diretório atual
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construir o caminho absoluto para o arquivo dados_processados.csv
file_path = os.path.join(current_dir, '../dados/processados/dados_processados.csv')


print("Carregando dados processados...")
dados = pd.read_csv(file_path)


print("Primeiras linhas do dataset:")
print(dados.head())

# Descrever o dataset
print("Descrição do dataset:")
print(dados.describe())

# Visualizar a distribuição da variável 'Amount'
print("Visualizando a distribuição de 'Amount'...")
plt.figure(figsize=(10, 6))
sns.histplot(dados['Amount'], bins=50)
plt.title("Distribuição do Valor da Transação ('Amount')")
plt.xlabel("Valor da Transação")
plt.ylabel("Frequência")
plt.savefig(os.path.join(current_dir, '../relatorios/figuras/distribuicao_amount.png'))
plt.show()

# Gráfico da Matriz de correlação
print("Calculando a matriz de correlação...")
matriz_correlacao = dados.corr()

plt.figure(figsize=(20, 16))  
sns.heatmap(matriz_correlacao, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"size": 10})
plt.title("Matriz de Correlação entre as Variáveis", fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig(os.path.join(current_dir, '../relatorios/figuras/matriz_correlacao.png'))
plt.show()
