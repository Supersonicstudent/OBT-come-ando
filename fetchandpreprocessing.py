from pprint import pprint
import googlemaps 
import requests
from PIL import Image
from io import BytesIO
from datetime import datetime
import pandas as pd 
import json
import ee
import geemap
import streamlit as st
import folium
import time
from selenium import webdriver
from flask import Flask

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
                                     departure_time=datetime(year=2024, month=6, day=26, hour=20, minute=0).timestamp() )
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
    
import ee
import geemap
from flask import Flask
import time
from selenium import webdriver
from PIL import Image

# Inicializar a biblioteca Earth Engine
ee.Initialize()

# Definir a região de interesse (uma área menor para teste, por exemplo)
region = ee.Geometry.Rectangle([-122.45, 37.74, -122.4, 37.8])

# Adicionar a camada de imagem de satélite do Earth Engine
# A camada Landsat 8 TOA Reflectance (LANDSAT/LC08/C01/T1_RT_TOA) é usada como exemplo
image = ee.ImageCollection('LANDSAT/LC08/C01/T1_RT_TOA').median().clip(region)

# Exportar a imagem como um GeoTIFF
export_task = ee.batch.Export.image.toDrive(
    image=image,
    description='Landsat_Export',
    folder='EarthEngineImages',
    scale=30,
    region=region.getInfo()['coordinates'],
    fileFormat='GeoTIFF'
)

export_task.start()

# Criar um mapa interativo usando geemap
Map = geemap.Map(center=[0, 0], zoom=2)

# Adicionar a imagem ao mapa
Map.addLayer(image, {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 0.3, 'gamma': 1.4}, 'Landsat 8')

# Adicionar controles de camada ao mapa
Map.addLayerControl()

# Salvar o mapa como um arquivo HTML
Map.save('map.html')

# Usar selenium para capturar uma imagem do mapa
def save_map_as_image(html_file, output_image):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1200,800")
    driver = webdriver.Chrome(options=options)
    driver.get(f"file:///{html_file}")
    time.sleep(10)  # Aumentar o tempo de espera para garantir o carregamento completo do mapa
    driver.save_screenshot(output_image)
    driver.quit()

# Salvar o mapa como uma imagem PNG
save_map_as_image('map.html', 'map.png')

# Abrir a imagem salva e exibir
img = Image.open('map.png')
img.show()

app = Flask(__name__)

@app.route('/')
def index():
    return open('map.html').read()

if __name__ == '__main__':
    app.run(debug=True)
