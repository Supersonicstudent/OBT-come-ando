import pandas as pd
import random

# Carregar o DataFrame
csv_file_path = r'C:\Users\steve\OneDrive\Área de Trabalho\OBT\OBT\OBT\extracted\traffic.csv'
df = pd.read_csv(csv_file_path)

# Adicionar coordenadas geográficas fictícias para cada junção
junction_coordinates = {
    1: (40.7128, -74.0060),  # Exemplo de coordenada
    2: (34.0522, -118.2437),
    3: (41.8781, -87.6298),
    4: (51.5074, -0.1278)
}

df['latitude'] = df['Junction'].apply(lambda x: junction_coordinates[x][0])
df['longitude'] = df['Junction'].apply(lambda x: junction_coordinates[x][1])

# Definir níveis de congestionamento com base no número de veículos
def categorize_congestion(vehicles):
    if vehicles < 10:
        return 'Baixo'
    elif vehicles < 30:
        return 'Médio'
    else:
        return 'Alto'

df['congestion_level'] = df['Vehicles'].apply(categorize_congestion)

# Adicionar colunas fictícias de tipo de melhoria e taxa de sucesso
improvement_types = ['Adicionar faixa extra', 'Melhorar sinalização', 'Implementar semáforos inteligentes']
success_rates = [0.7, 0.5, 0.8]  # Exemplos de taxas de sucesso

df['improvement_type'] = [random.choice(improvement_types) for _ in range(len(df))]
df['success_rate'] = [random.choice(success_rates) for _ in range(len(df))]

print(df.head())

# Salvar o DataFrame enriquecido em um novo arquivo CSV
df.to_csv(r'C:\Users\steve\OneDrive\Área de Trabalho\OBT\OBT\OBT\enriched_traffic_data.csv', index=False)
import pandas as pd

# Carregar o DataFrame original
csv_file_path = r'C:\Users\steve\OneDrive\Área de Trabalho\OBT\OBT\OBT\extracted\traffic.csv'
df_original = pd.read_csv(csv_file_path)

# Definir as coordenadas fictícias para cada junção
junction_coordinates = {
    1: (40.7128, -74.0060),
    2: (34.0522, -118.2437),
    3: (41.8781, -87.6298),
    4: (51.5074, -0.1278)
}

# Verificar se as junções no DataFrame original correspondem às chaves do dicionário junction_coordinates
original_junctions = df_original['Junction'].unique()
expected_junctions = list(junction_coordinates.keys())

# Verificar se todas as junções esperadas estão presentes no DataFrame original
if set(expected_junctions).issubset(set(original_junctions)):
    print("Todas as junções esperadas estão presentes no DataFrame original.")
else:
    print("Alguns dados de junção estão faltando ou não correspondem às junções esperadas.")
