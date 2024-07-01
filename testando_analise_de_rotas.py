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
    
    # Imprimir a duração original e a duração em tráfego para cada passo
    for step_duration, step_duration_in_traffic in step_durations:
        print(f"Duração do passo: {step_duration / 60:.2f} minutos, Duração em tráfego: {step_duration_in_traffic / 60:.2f} minutos")

# Convertendo os dados para um DataFrame
steps_df = pd.DataFrame(steps_data)

# Exibir o DataFrame
print(steps_df)
diference_of_times_list = []
for step_duration, step_duration_in_traffic in step_durations:
    diference_of_times = step_duration_in_traffic/60 - step_duration/60
    diference_of_times_list.append(diference_of_times)
def print_maior_diferenca(difference_of_times_list):
    # Verifica se a lista não está vazia
    if difference_of_times_list:
        # Encontra o maior valor na lista
        maior_diferenca = max(difference_of_times_list)
        print(f"Maior diferença de tempo: {maior_diferenca:.2f} minutos")
    else:
        print("A lista de diferenças está vazia.")
print_maior_diferenca(diference_of_times_list)