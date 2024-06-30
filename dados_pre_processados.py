import googlemaps 
import requests
from datetime import datetime
import pandas as pd 
import json
import streamlit as st

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
            st.error(f"Erro na solicitação da rota: {directions_data['status']}")
            return None
    else:
        st.error(f"Erro ao obter a rota. Código de status: {response.status_code}")
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
        st.error(f"Erro ao obter a imagem do mapa. Código de status: {response.status_code}")
        return None

api_key = "AIzaSyDdTREWbb7NJRvkBjReLpRdgNIyqJeLcbM"

st.title("Mapa de Trajeto")

origin = st.text_input("Origem", "Brasília, Brazil")
destination = st.text_input("Destino", "Valparaíso de Goiás, Brazil")

if st.button("Obter Rota"):
    routes = get_directions(api_key, origin, destination)
    
    if routes:
        route_option = st.selectbox("Selecione uma rota", range(len(routes)))
        route_map = get_route_map(api_key, routes[route_option])
        
        if route_map:
            st.image(route_map, caption="Mapa do Trajeto")
