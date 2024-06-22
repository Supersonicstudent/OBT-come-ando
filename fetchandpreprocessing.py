from pprint import pprint
import googlemaps 
import requests
from PIL import Image
from io import BytesIO
from datetime import datetime
import pandas as pd 
import json

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
directions_result = gmaps.directions("Sydney Town Hall",
                                     "Parramatta, NSW",
                                     mode="driving",
                                     departure_time=now)
# Convert the result of the directions request to JSON
json_data = json.dumps(directions_result)
data = json.loads(json_data)

# Access the 'legs' list where distance information is stored
legs = data[0]['legs']  # Corrigido para acessar o primeiro item da lista de direções

# Create an empty list to store distances
distances = []

# Loop through each leg and extract the distance information
for leg in legs:
    distance_dict = leg['distance']
    distance_text = distance_dict['text']
    distances.append(distance_text)

# Create a pandas Series from the list of distances
distance_series = pd.Series(distances)

# Print the Series containing just the distances
print(distance_series)



