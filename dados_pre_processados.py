import googlemaps 
import requests
from datetime import datetime
import pandas as pd 
import json
import streamlit as st
from PIL import Image, ImageOps
from io import BytesIO

# Configurações básicas do API de direções
api_key = 'AIzaSyDdTREWbb7NJRvkBjReLpRdgNIyqJeLcbM'
gmaps = googlemaps.Client(key=api_key)
# Request directions via driving
origem_geral = "Santa Maria, Distrito Federal, Brazil"  # Estabelece uma origem para ambos os API´s
destino_geral = "Valparaíso de Goiás, Brazil" # Estabelece um destino para ambos os API´s
now = datetime.now()
directions_result = gmaps.directions(origem_geral,
                                     destino_geral,
                                     mode="driving",
                                     departure_time=datetime(year=2024, month=7, day=1, hour=19, minute=0).timestamp() )
# Convert the result of the directions request to JSON
json_data = json.dumps(directions_result)
data = json.loads(json_data)

# Access the 'legs' list where duration information is stored
legs = data[0]['legs']  # Corrigido para acessar o primeiro item da lista de durações

# Create an empty list to store durations
durations = ["duration: "]
duration_in_traffic = ["duration in traffic: "]

# Loop through each leg and extract the duration information
for leg in legs:
    duration_dict = leg['duration']
    duration_text = duration_dict['text']
    durations.append(duration_text)
for leg in legs:
    duration_in_traffic_dict = leg['duration_in_traffic']
    duration_in_traffic_text = duration_in_traffic_dict['text']
    duration_in_traffic.append(duration_in_traffic_text)
# Create a pandas Series from the list of durations
durations_series = pd.Series(durations)
duration_in_traffic_series = pd.Series(duration_in_traffic)

# Print the Series containing just the durations
print(durations_series)
print(duration_in_traffic_series)

# Parte do mapa com o trajeto 
def get_directions(api_key, origin, destination):
    directions_url = "https://maps.googleapis.com/maps/api/directions/json?"

    directions_params = {
        "origin": origin,
        "destination": destination,
        "key": api_key,
        "alternatives": "true"
    }

    response = requests.get(directions_url, params=directions_params)
    
    if response.status_code == 200:
        directions_data = response.json()
        if directions_data["status"] == "OK":
            routes = [route["overview_polyline"]["points"] for route in directions_data["routes"]]
            return routes
        else:
            print(f"Erro na solicitação da rota: {directions_data['status']}")
            return None
    else:
        print(f"Erro ao obter a rota. Código de status: {response.status_code}")
        return None

def get_route_map(api_key, route, size="600x300", maptype="satellite", weight=2, color="0x0000FF"):
    static_map_url = "https://maps.googleapis.com/maps/api/staticmap?"

    static_map_params = {
        "size": size,
        "maptype": maptype,
        "key": api_key,
        "path": f"color:{color}|weight:{weight}|enc:{route}"
    }

    response = requests.get(static_map_url, params=static_map_params)

    if response.status_code == 200:
        return response.content
    else:
        print(f"Erro ao obter a imagem do mapa. Código de status: {response.status_code}")
        return None

def save_image(api_key, origin, destination, file_path):
    routes = get_directions(api_key, origin, destination)
    
    if routes:
        route_map = get_route_map(api_key, routes[0])  # Aqui pegamos apenas a primeira rota
        if route_map:
            with open(file_path, 'wb') as f:
                f.write(route_map)
            print(f"Imagem salva em: {file_path}")
        else:
            print("Não foi possível obter o mapa do trajeto.")
    else:
        print("Não foi possível obter as rotas.")
# Exemplo de uso:
api_key = "AIzaSyDdTREWbb7NJRvkBjReLpRdgNIyqJeLcbM"
origin = origem_geral
destination = destino_geral
# Montando o nome do arquivo com formatação de string
nome_arquivo = f"Rota de {origem_geral} para {destino_geral}.jpg"
save_image(api_key, origin, destination, nome_arquivo)