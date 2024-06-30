import requests
import streamlit as st

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

origin = st.text_input("Origem", "Floripa, Brazil")
destination = st.text_input("Destino", "Blumenau, Brazil")

if st.button("Obter Rota"):
    routes = get_directions(api_key, origin, destination)
    
    if routes:
        route_option = st.selectbox("Selecione uma rota", range(len(routes)))
        route_map = get_route_map(api_key, routes[route_option])
        
        if route_map:
            st.image(route_map, caption="Mapa do Trajeto")

