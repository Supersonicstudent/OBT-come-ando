import googlemaps
import requests
from datetime import datetime
import pandas as pd
import json
from PIL import Image
from io import BytesIO
import torch
import torchvision.transforms as transforms
import torchvision.models.segmentation as segmentation
import numpy as np
import warnings
import os
from flask import Flask, request

# Configurações básicas do API de direções
api_key = 'AIzaSyDdTREWbb7NJRvkBjReLpRdgNIyqJeLcbM'
gmaps = googlemaps.Client(key=api_key)

# Função para obter direções e calcular tempos de duração e tráfego
def get_directions_and_calculate_times(origem, destino):
    directions_result = gmaps.directions(
        origem,
        destino,
        mode="driving",
        departure_time=datetime.now().timestamp()
    )

    data = pd.json_normalize(directions_result, 'legs')

    for route in directions_result:
        for leg in route['legs']:
            total_duration = leg['duration']['value']
            total_duration_in_traffic = leg.get('duration_in_traffic', {}).get('value', total_duration)

    traffic_difference = total_duration_in_traffic - total_duration

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
                'duration': step_duration / 60,
                'duration_in_traffic': step_duration_in_traffic / 60
            }
            steps_data.append(step_data)

            diference_of_times = step_duration_in_traffic / 60 - step_duration / 60
            diference_of_times_list.append((diference_of_times, step['start_location'], step['end_location']))

    steps_df = pd.DataFrame(steps_data)

    max_diference = max(diference_of_times_list, key=lambda x: x[0])
    start_location_var = max_diference[1]
    end_location_var = max_diference[2]

    start_address = gmaps.reverse_geocode((start_location_var['lat'], start_location_var['lng']))[0]['formatted_address']
    end_address = gmaps.reverse_geocode((end_location_var['lat'], end_location_var['lng']))[0]['formatted_address']

    return directions_result, start_location_var, end_location_var, start_address, end_address

# Função para identificar junções
def identify_junctions(directions_result, start_location, end_location):
    junctions = []
    recording = False

    for step in directions_result[0]['legs'][0]['steps']:
        if step['start_location'] == start_location:
            recording = True
        if recording and 'maneuver' in step:
            if 'roundabout' in step['maneuver'] or 'merge' in step['maneuver'] or 'fork' in step['maneuver']:
                junctions.append(step['end_location'])

    return junctions

# Função para sugerir mudanças com base na segmentação
def suggest_changes(junctions):
    improvements = []
    if junctions:
        for junction in junctions:
            improvements.append({
                'location': {'lat': junction['lat'], 'lng': junction['lng']},
                'suggestion': 'adicionar um semáforo'
            })
    else:
        improvements.append({
            'location': None,
            'suggestion': 'adicionar faixa'
        })
    return improvements

# Função para converter coordenadas em endereços usando Google Maps Reverse Geocoding
def convert_coords_to_addresses(coords):
    addresses = []
    for coord in coords:
        lat = float(coord['lat'])
        lng = float(coord['lng'])

        try:
            reverse_geocode_result = gmaps.reverse_geocode((lat, lng))
            if reverse_geocode_result:
                address = reverse_geocode_result[0]['formatted_address']
            else:
                reverse_geocode_result = gmaps.reverse_geocode(latlng=(lat, lng), location_type='ROOFTOP')
                if reverse_geocode_result:
                    address = reverse_geocode_result[0]['formatted_address']
                else:
                    address = "Endereço não encontrado"
        except googlemaps.exceptions.ApiError as api_err:
            address = f"Erro na API do Google Maps: {str(api_err)}"
        except googlemaps.exceptions.TransportError as transport_err:
            address = f"Erro de transporte na solicitação: {str(transport_err)}"
        except googlemaps.exceptions.Timeout as timeout_err:
            address = f"Tempo esgotado na solicitação: {str(timeout_err)}"
        except Exception as e:
            address = f"Erro ao buscar endereço: {str(e)}"

        addresses.append(address)
    return addresses

# Função para processar dados de rotas e gerar imagem do mapa
def save_image(api_key, origin, destination, file_path, center, zoom):
    directions_result, start_location_var, end_location_var, start_address, end_address = get_directions_and_calculate_times(origin, destination)
    routes = get_directions(api_key, origin, destination)

    if routes:
        route_map = get_route_map(api_key, routes[0], center=center, zoom=zoom)
        if route_map:
            with open(file_path, 'wb') as f:
                f.write(route_map)
            print(f"Imagem salva em: {file_path}")
        else:
            print("Não foi possível obter o mapa do trajeto.")
    else:
        print("Não foi possível obter as rotas.")

# Função para retornar o resultado final
def return_result(start_address, end_address, improvements):
    for improvement in improvements:
        if improvement['location']:
            coords = improvement['location']
            addresses = convert_coords_to_addresses([coords])
            improvement['address'] = addresses[0]

    engarrafamento_message = f'O engarrafamento na viagem especificada começa no trecho: {start_address} e termina em: {end_address}.'
    if improvements[0]['location'] is None:
        return engarrafamento_message + " Para resolver tal problema, o Mapper.AI recomenda a avaliação da viabilidade de adicionar uma faixa na referida via"
    else:
        sugestao = improvements[0]['suggestion']
        coords = improvements[0]['location']
        address = improvements[0]['address']
        return f'{engarrafamento_message} Para resolver tal problema, o Mapper.AI recomenda a avaliação da viabilidade de {sugestao} no local: {coords} que corresponde a {address}.'

app = Flask(__name__)

@app.route('/process_data', methods=['POST'])
def process_data():
    origem_geral = request.form.get('origem_geral')
    destino_geral = request.form.get('destino_geral')

    directions_result, start_location_var, end_location_var, start_address, end_address = get_directions_and_calculate_times(origem_geral, destino_geral)
    junctions = identify_junctions(directions_result, start_location_var, end_location_var)
    improvements = suggest_changes(junctions)
    result_message = return_result(start_address, end_address, improvements)

    return result_message

if __name__ == '__main__':
    app.run(debug=True)
