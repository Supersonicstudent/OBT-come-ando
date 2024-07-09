import googlemaps 
import requests
from datetime import datetime
import pandas as pd 
import warnings

# Configurações básicas do API de direções
api_key = 'AIzaSyDdTREWbb7NJRvkBjReLpRdgNIyqJeLcbM'  # Substitua pelo seu API key real
gmaps = googlemaps.Client(key=api_key)

# Suprimir avisos do PyTorch
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

def analyze_traffic(origem_geral, destino_geral):
    # Realize a solicitação de direções via condução
    directions_result = gmaps.directions(
        origem_geral,
        destino_geral,
        mode="driving",
        departure_time=datetime.now().timestamp()
    )

    # Converter o resultado da solicitação de direções para um DataFrame do pandas
    data = pd.json_normalize(directions_result, 'legs')

    # Obter a duração total e a duração em tráfego
    for route in directions_result:
        for leg in route['legs']:
            total_duration = leg['duration']['value']
            total_duration_in_traffic = leg.get('duration_in_traffic', {}).get('value', total_duration)

    # Calcular a diferença total de duração em tráfego
    traffic_difference = total_duration_in_traffic - total_duration

    # Acessar os dados de 'duration' dentro do DataFrame e calcular a duração em tráfego para cada passo
    steps_data = []
    diference_of_times_list = []

    for index, row in data.iterrows():
        total_step_duration = sum(step['duration']['value'] for step in row['steps'])

        for step in row['steps']:
            step_duration = step['duration']['value']
            proportion_of_total = step_duration / total_step_duration
            step_duration_in_traffic = step_duration + (proportion_of_total * traffic_difference)

            step_data = {
                'start_location': step['start_location'],
                'end_location': step['end_location'],
                'duration': step_duration / 60,  # Convertendo para minutos
                'duration_in_traffic': step_duration_in_traffic / 60  # Convertendo para minutos
            }
            steps_data.append(step_data)

            diference_of_times = step_duration_in_traffic / 60 - step_duration / 60
            diference_of_times_list.append((diference_of_times, step['start_location'], step['end_location']))

    # Convertendo os dados para um DataFrame
    steps_df = pd.DataFrame(steps_data)

    # Encontrar a maior diferença e armazenar as localizações
    max_diference = max(diference_of_times_list, key=lambda x: x[0])
    start_location_max = max_diference[1]
    end_location_max = max_diference[2]

    # Converter coordenadas para endereços
    start_address = gmaps.reverse_geocode((start_location_max['lat'], start_location_max['lng']))[0]['formatted_address']
    end_address = gmaps.reverse_geocode((end_location_max['lat'], end_location_max['lng']))[0]['formatted_address']

    # Função para identificar junções
    def identify_junctions(directions_result):
        junctions = []
        # Percorre as etapas das direções para identificar junções
        for step in directions_result[0]['legs'][0]['steps']:
            if 'maneuver' in step:
                if 'roundabout' in step['maneuver'] or 'merge' in step['maneuver'] or 'fork' in step['maneuver']:
                    junctions.append(step['end_location'])            
        return junctions

    return start_address, end_address, directions_result

# Função para retornar os endereços
def get_congestion_addresses(origem_geral, destino_geral):
    start_address, end_address, directions_result = analyze_traffic(origem_geral, destino_geral)
    return start_address, end_address, directions_result
