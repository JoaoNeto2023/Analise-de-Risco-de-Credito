import pandas as pd
import os

'''@Autor: João Neto
   Data: 04/07/2024'''

# Obter o caminho absoluto do diretório atual
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construir o caminho absoluto para o arquivo dataset.csv
file_path = os.path.join(current_dir, '../dados/brutos/dataset.csv')

print("Carregando dados...")
# Carregar os dados brutos
dados = pd.read_csv(file_path)
#Salva todos os dados
processed_path = os.path.join(current_dir, '../dados/processados/dados_processados.csv')
dados.to_csv(processed_path, index=False)

print(f"Dados processados salvos em: {processed_path}")
