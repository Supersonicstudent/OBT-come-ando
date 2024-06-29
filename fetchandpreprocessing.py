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
                                     departure_time=datetime(year=2024, month=6, day=29, hour=10, minute=0).timestamp() )
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



# Defina suas credenciais e a URL da API
INSTANCE_ID = '7408c9a5-3710-4be7-a186-4b21a2c07c57' 
LAYER = 'CAMADA-QUE-EU-QUERO'  # Nome da camada conforme configurado no Sentinel Hub
BBOX = '-47.9811389,-16.0499444,-47.9753611,-16.0505556'
WIDTH = 512
HEIGHT = 512
FORMAT = 'image/png'

# URL para solicitação
url = f"https://services.sentinel-hub.com/ogc/wms/{INSTANCE_ID}?REQUEST=GetMap&BBOX={BBOX}&LAYERS={LAYER}&WIDTH={WIDTH}&HEIGHT={HEIGHT}&FORMAT={FORMAT}"

# Faça a solicitação
response = requests.get(url)

# Verifique se a solicitação foi bem-sucedida
if response.status_code == 200:
    image = Image.open(BytesIO(response.content))
    image.show()  # Exibe a imagem
    image.save('satellite_image.png')  # Salva a imagem
else:
    print("Erro ao obter a imagem:", response.status_code, response.text)



