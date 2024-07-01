import googlemaps
import pandas as pd
from datetime import datetime

# Configurações básicas do API de direções
api_key = 'AIzaSyDdTREWbb7NJRvkBjReLpRdgNIyqJeLcbM'
gmaps = googlemaps.Client(key=api_key)

# Estabeleça uma origem e um destino para ambos os APIs
origem_geral = "Santa Maria, Distrito Federal, Brazil"
destino_geral = "Valparaíso de Goiás, Brazil"

# Realize a solicitação de direções via condução
directions_result = gmaps.directions(
    origem_geral,
    destino_geral,
    mode="driving",
    departure_time=datetime(year=2024, month=7, day=1, hour=19, minute=0).timestamp()
)

# Converter o resultado da solicitação de direções para um DataFrame do pandas
data = pd.json_normalize(directions_result, 'legs')

# Obter a duração total e a duração em tráfego
for route in directions_result:
    for leg in route['legs']:
        total_duration = leg['duration']['value']
        total_duration_in_traffic = leg.get('duration_in_traffic', {}).get('value', total_duration)
        print(f"Duração total: {total_duration / 60:.2f} mimutos")
        print(f"Duração total em tráfego: {total_duration_in_traffic / 60:.2f} minutos")

# Calcular a diferença total de duração em tráfego
traffic_difference = total_duration_in_traffic - total_duration

# Acessar os dados de 'duration' dentro do DataFrame e calcular a duração em tráfego para cada passo
steps_data = []

for index, row in data.iterrows():
    step_durations = []
    total_step_duration = sum(step['duration']['value'] for step in row['steps'])
    
    for step in row['steps']:
        step_duration = step['duration']['value']
        proportion_of_total = step_duration / total_step_duration
        step_duration_in_traffic = step_duration + (proportion_of_total * traffic_difference)
        
        step_data = {
            'start_location': step['start_location'],
            'end_location': step['end_location'],
            'duration': step_duration /60,
            'duration_in_traffic': step_duration_in_traffic /60
        }
        steps_data.append(step_data)
        step_durations.append((step_duration, step_duration_in_traffic))

# Convertendo os dados para um DataFrame
steps_df = pd.DataFrame(steps_data)
diference_of_times_list = []
for step_duration, step_duration_in_traffic in step_durations:
    diference_of_times = step_duration_in_traffic/60 - step_duration/60
    diference_of_times_list.append(diference_of_times)
def print_maior_diferenca(difference_of_times_list, steps_df):
    # Verifica se a lista não está vazia
    if difference_of_times_list:
        # Encontra o índice do maior valor na lista
        indice_maior_diferenca = difference_of_times_list.index(max(difference_of_times_list))
        
        # Acessa os dados do passo com maior diferença de tempo
        start_location = steps_df.loc[indice_maior_diferenca, 'start_location']
        end_location = steps_df.loc[indice_maior_diferenca, 'end_location']
        
        # Imprime o maior valor de diferença e as localizações
        maior_diferenca = difference_of_times_list[indice_maior_diferenca]
        print(f"Maior diferença de tempo: {maior_diferenca:.2f} minutos")
        print(f"Start Location: {start_location}")
        print(f"End Location: {end_location}")
    else:
        print("A lista de diferenças está vazia.")

# Exemplo de uso com o seu DataFrame steps_df e lista difference_of_times_list
# Supondo que você já tenha calculado difference_of_times_list e preenchido steps_df
# com os dados necessários.

# Chamada da função para imprimir o maior valor da lista
print_maior_diferenca(diference_of_times_list, steps_df)
