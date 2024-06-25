from pprint import pprint
import googlemaps 
import requests
from PIL import Image
from datetime import datetime
import pandas as pd 
import json
import ee
import geemap

def obter_imagem_mapa(api_key, location, zoom, width, height):
    url = 'https://maps.googleapis.com/maps/api/staticmap'
    params = {
        'center': location,
        'zoom': zoom,
        'size': f"{width}x{height}",
        'key': api_key
    }
    response = requests.get(url, params=params)
    image = Image.open(BytesIO(response.content))
    return image

# Exemplo de utilização
api_key = 'AIzaSyDdTREWbb7NJRvkBjReLpRdgNIyqJeLcbM'
location = 'Quadra E, Rua A, Valparaíso de Goiás, Goiás, Brazil'
zoom = 20 # Tente um nível de zoom mais baixo para uma área maior
imagem_mapa = obter_imagem_mapa(api_key, location, zoom, width=2048, height=1366)
imagem_mapa.show()

gmaps = googlemaps.Client(key=api_key)

# Geocoding an address
geocode_result = gmaps.geocode('1600 Amphitheatre Parkway, Mountain View, CA')

# Look up an address with reverse geocoding
reverse_geocode_result = gmaps.reverse_geocode((40.714224, -73.961452))

# Request directions via driving
now = datetime.now()
directions_result = gmaps.directions("Brasília, Brazil",
                                     "Valparaíso de Goiás, Goiás",
                                     mode="driving",
                                     departure_time=datetime(year=2024, month=6, day=25, hour=19, minute=0).timestamp() )
# Convert the result of the directions request to JSON
json_data = json.dumps(directions_result)
data = json.loads(json_data)

# Access the 'legs' list where distance information is stored
legs = data[0]['legs']  # Corrigido para acessar o primeiro item da lista de direções

# Create an empty list to store distances
durations = ["duration: "]
duration_in_traffic = ["duration in traffic: "]

# Loop through each leg and extract the distance information
for leg in legs:
    duration_dict = leg['duration']
    duration_text = duration_dict['text']
    durations.append(duration_text)
for leg in legs:
    duration_in_traffic_dict = leg['duration_in_traffic']
    duration_in_traffic_text = duration_in_traffic_dict['text']
    duration_in_traffic.append(duration_in_traffic_text)
# Create a pandas Series from the list of distances
durations_series = pd.Series(durations)
duration_in_traffic_series = pd.Series(duration_in_traffic)

# Print the Series containing just the distances
print(durations_series)
print(duration_in_traffic_series)

import ee

# ID do seu projeto no Google Cloud
project_id = 'protean-trilogy-423518-b9'  # Substitua pelo seu ID de projeto

# Inicializar a API especificando o projeto
try:
    ee.Initialize(project=project_id)
    print("Google Earth Engine Initialized successfully!")
except ee.EEException as e:
    print(f"Erro ao inicializar o Google Earth Engine: {e}")
    
# Criar um mapa interativo
Map = geemap.Map(center=[0, 0], zoom=2)

# Adicionar a camada de imagem de satélite do Earth Engine
# A camada Landsat 8 TOA Reflectance (LANDSAT/LC08/C01/T1_RT_TOA) é usada como exemplo
image = ee.ImageCollection('LANDSAT/LC08/C01/T1_RT_TOA').median()

# Definir parâmetros de visualização para a imagem
vis_params = {
    'bands': ['B4', 'B3', 'B2'],
    'min': 0,
    'max': 0.3,
    'gamma': 1.4,
}

# Adicionar a imagem ao mapa
Map.addLayer(image, vis_params, 'Landsat 8')

# Exibir o mapa
Map.addLayerControl()  # Adicionar controles de camada ao mapa
Map








