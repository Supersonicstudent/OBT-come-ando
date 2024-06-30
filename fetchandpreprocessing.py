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
from sentinelhub import SHConfig, BBox, CRS, MimeType, SentinelHubRequest, DataCollection, bbox_to_dimensions
import numpy as np 



# Exemplo de utilização
api_key = 'AIzaSyDdTREWbb7NJRvkBjReLpRdgNIyqJeLcbM'
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
                                     departure_time=datetime(year=2024, month=7, day=1, hour=19, minute=0).timestamp() )
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

# Configurações da API
config = SHConfig()

config.instance_id = '7408c9a5-3710-4be7-a186-4b21a2c07c57'
config.sh_client_id = '7e35353d-f06b-493f-a8d9-3bfafa409142'
config.sh_client_secret = 'Z9Xn6GlcnZXcMqsjGdso7nu7aitDeQJ7'


# Configuração da bounding box
# Use coordenadas da área de interesse
bounding_box = BBox(bbox=[-47.9292, -15.7801, -47.9291, -15.7800], crs=CRS.WGS84)  # Ajuste essas coordenadas para sua área de interesse
resolution = 2  # Resolução em metros por pixel

# Calcular dimensões da imagem
width, height = bbox_to_dimensions(bounding_box, resolution)

# Se a largura ou altura for maior que 2500, dividir a bounding box
max_size = 2500
num_tiles_x = int(np.ceil(width / max_size))
num_tiles_y = int(np.ceil(height / max_size))

# Configuração do Sentinel Hub
config = SHConfig()
config.instance_id = '7408c9a5-3710-4be7-a186-4b21a2c07c57'
config.sh_client_id = '49ccec93-d9b0-4416-b408-f7f612c7445b'
config.sh_client_secret = ''

def get_image_for_bbox(bbox):
    request = SentinelHubRequest(
        evalscript='''
            //VERSION=3
            function setup() {
                return {
                    input: ["B04", "B03", "B02"],
                    output: { bands: 3 }
                };
            }

            function evaluatePixel(sample) {
                let minVal = 0.0;
                let maxVal = 0.3;
                let viz = [
                    sample.B04 > maxVal ? 1.0 : sample.B04 < minVal ? 0.0 : (sample.B04 - minVal) / (maxVal - minVal),
                    sample.B03 > maxVal ? 1.0 : sample.B03 < minVal ? 0.0 : (sample.B03 - minVal) / (maxVal - minVal),
                    sample.B02 > maxVal ? 1.0 : sample.B02 < minVal ? 0.0 : (sample.B02 - minVal) / (maxVal - minVal),
                ];
                return viz;
            }
        ''',
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=('2023-06-01', '2023-06-30')
            )
        ],
        responses=[
            SentinelHubRequest.output_response('default', MimeType.PNG)
        ],
        bbox=bbox,
        size=bbox_to_dimensions(bbox, resolution),
        config=config,
        data_folder=r'C:\Users\steve\OneDrive\Área de Trabalho\OBT\OBT\OBT'
    )

    return request.get_data(save_data=True)[0]

# Dividir a bounding box e obter imagens
for i in range(num_tiles_x):
    for j in range(num_tiles_y):
        tile_bbox = BBox([
            bounding_box.min_x + i * (bounding_box.max_x - bounding_box.min_x) / num_tiles_x,
            bounding_box.min_y + j * (bounding_box.max_y - bounding_box.min_y) / num_tiles_y,
            bounding_box.min_x + (i + 1) * (bounding_box.max_x - bounding_box.min_x) / num_tiles_x,
            bounding_box.min_y + (j + 1) * (bounding_box.max_y - bounding_box.min_y) / num_tiles_y
        ], crs=bounding_box.crs)
        
        image = get_image_for_bbox(tile_bbox)

        # Visualizar imagem
        import matplotlib.pyplot as plt
        plt.imshow(image)
        plt.title(f'Tile {i}, {j}')
        plt.show()


