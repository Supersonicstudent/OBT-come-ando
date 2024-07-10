import googlemaps 
import requests
from datetime import datetime
import pandas as pd 
import json
import streamlit as st
from PIL import Image, ImageOps
from io import BytesIO
from servidor_flask import origem_geral, destino_geral

# Configurações básicas do API de direções
api_key = 'AIzaSyDdTREWbb7NJRvkBjReLpRdgNIyqJeLcbM'
gmaps = googlemaps.Client(key=api_key)

# Realize a solicitação de direções via condução
directions_result = gmaps.directions(
    origem_geral,
    destino_geral,
    mode="driving",
    departure_time=datetime(year=2024, month=7, day=10, hour=19, minute=0).timestamp()
)

# Converter o resultado da solicitação de direções para um DataFrame do pandas
data = pd.json_normalize(directions_result, 'legs')

# Obter a duração total e a duração em tráfego
for route in directions_result:
    for leg in route['legs']:
        total_duration = leg['duration']['value']
        total_duration_in_traffic = leg.get('duration_in_traffic', {}).get('value', total_duration)
        print(f"Duração total: {total_duration / 60:.2f} minutos")
        print(f"Duração total em tráfego: {total_duration_in_traffic / 60:.2f} minutos")

# Calcular a diferença total de duração em tráfego
traffic_difference = total_duration_in_traffic - total_duration

# Acessar os dados de 'duration' dentro do DataFrame e calcular a duração em tráfego para cada passo
steps_data = []
diference_of_times_list = []
step_durations = []

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
        step_durations.append((step_duration, step_duration_in_traffic))
    
        diference_of_times = step_duration_in_traffic / 60 - step_duration / 60
        diference_of_times_list.append((diference_of_times, step['start_location'], step['end_location']))

# Convertendo os dados para um DataFrame
steps_df = pd.DataFrame(steps_data)

# Encontrar a maior diferença e armazenar as localizações
max_diference = max(diference_of_times_list, key=lambda x: x[0])
max_diference_value = max_diference[0]
start_location_max = max_diference[1]
end_location_max = max_diference[2]

# Armazenando as localizações em variáveis para uso posterior
start_location_var = start_location_max
end_location_var = end_location_max

# Converter coordenadas para endereços
start_address = gmaps.reverse_geocode((start_location_var['lat'], start_location_var['lng']))[0]['formatted_address']
end_address = gmaps.reverse_geocode((end_location_var['lat'], end_location_var['lng']))[0]['formatted_address']

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

# Corrigido para passar directions_result em vez de data
identify_junctions(directions_result, start_location_var, end_location_var)

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

def get_route_map(api_key, route, center, zoom, size="600x300", maptype="satellite", weight=2, color="0x0000FF"):
    static_map_url = "https://maps.googleapis.com/maps/api/staticmap?"
    static_map_params = {
        "size": size,
        "maptype": maptype,
        "key": api_key,
        "path": f"color:{color}|weight:{weight}|enc:{route}",
        "zoom": zoom,
        "center": center
    }
    response = requests.get(static_map_url, params=static_map_params)
    if response.status_code == 200:
        return response.content
    else:
        print(f"Erro ao obter a imagem do mapa. Código de status: {response.status_code}")
        return None

def save_image(api_key, origin, destination, file_path, center, zoom):
    routes = get_directions(api_key, origin, destination)
    
    if routes:
        route_map = get_route_map(api_key, routes[0], center=center, zoom=zoom)  # Aqui pegamos apenas a primeira rota
        if route_map:
            with open(file_path, 'wb') as f:
                f.write(route_map)
            print(f"Imagem salva em: {file_path}")
        else:
            print("Não foi possível obter o mapa do trajeto.")
    else:
        print("Não foi possível obter as rotas.")

# Exemplo de uso:
origin = start_address
destination = end_address
nome_arquivo = "Local do engarrafamento.jpg"
center = f"{start_location_var['lat']},{start_location_var['lng']}"
zoom = 18
file_path = r"C:\Users\steve\OneDrive\Documents\Mapper.AI\OBT-come-ando\{}".format(nome_arquivo)
save_image(api_key, origin, destination, file_path, center, zoom)

# Função para retornar os endereços
def get_congestion_cords():
    return start_location_var, end_location_var
def get_congestion_addresses():
    return start_address, end_address


import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.models.segmentation as segmentation
import numpy as np
import warnings
import googlemaps
import gmaps

# Suprimindo avisos de output indesejados
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

# Caminhos para os arquivos de entrada e saída
test_image_path = r'C:\Users\steve\OneDrive\Documents\Mapper.AI\OBT-come-ando\Local do engarrafamento.jpg'
checkpoint_path = r'C:\Users\steve\OneDrive\Área de Trabalho\trained_model.pth'
output_segmented_path = r'C:\Users\steve\OneDrive\Documents\Mapper.AI\OBT-come-ando\Segmentada_Salva.png'

# Transformações para a imagem de teste
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Carregando o modelo DeepLabV3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = segmentation.deeplabv3_resnet101(pretrained=False, num_classes=21).to(device)

# Carregando os pesos do modelo treinado
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
else:
    raise FileNotFoundError(f'O arquivo {checkpoint_path} não foi encontrado.')

# Função para realizar a segmentação da imagem
def segment_image(image_path, model, output_path):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)['out']
        output = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    segmented_image = Image.fromarray((output * 255).astype(np.uint8))
    segmented_image.save(output_path)
    return output

# Segmentando a imagem de teste
segmented_output = segment_image(test_image_path, model, output_segmented_path)

# Exibindo a imagem original e a segmentada
def display_images(original_image_path, segmented_image):
    original_image = Image.open(original_image_path)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title('Imagem Original')
    plt.imshow(original_image)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title('Resultado da Segmentação')
    plt.imshow(segmented_image, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Exemplo de uso
display_images(test_image_path, segmented_output)
improvements = []

# Função para sugerir mudanças com base na segmentação
def suggest_changes(junctions):
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
    gmaps = googlemaps.Client(key='AIzaSyDdTREWbb7NJRvkBjReLpRdgNIyqJeLcbM')  # Substitua com sua chave de API do Google Maps
    addresses = []
    
    for coord in coords:
        lat = float(coord['lat'])
        lng = float(coord['lng'])
        
        try:
            reverse_geocode_result = gmaps.reverse_geocode((coord['lat'],coord['lng']))
            if reverse_geocode_result:
                address = reverse_geocode_result[0]['formatted_address']
            else:
                # Tentativa adicional com diferentes parâmetros
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

# Exemplo de uso
start_location, end_location = get_congestion_cords()
junctions = identify_junctions(directions_result, start_location, end_location)
improvements = suggest_changes(junctions)

# Convertendo coordenadas para endereços
for improvement in improvements:
    if improvement['location']:
        coords = improvement['location']
        addresses = convert_coords_to_addresses([coords])
        improvement['address'] = addresses[0]


# Função para retornar o resultado final como uma string
def print_result(start_address, end_address, improvements):
    result_message = f'O engarrafamento na viagem especificada começa no trecho: {start_address} e termina em: {end_address}.\n'
    suggestions_dict = {}
    
    for improvement in improvements:
        suggestion = improvement['suggestion']
        location = improvement['location']
        address = improvement.get('address', 'Coordenadas não puderam ser convertidas em endereço')
        
        if suggestion not in suggestions_dict:
            suggestions_dict[suggestion] = []
        
        if location:
            suggestions_dict[suggestion].append((location, address))
    
    for suggestion, locations in suggestions_dict.items():
        if locations:
            if len(locations) == 1:
                if locations[0][0] is None:
                    result_message += f'Para resolver tal problema, o Mapper.AI recomenda a avaliação da viabilidade de {suggestion}.\n'
                else:
                    loc = locations[0][0]
                    addr = locations[0][1]
                    result_message += f'Para resolver tal problema, o Mapper.AI recomenda a avaliação da viabilidade de {suggestion} no seguinte local: Latitude: {loc["lat"]}, Longitude: {loc["lng"]} que corresponde a {addr}.\n'
            else:
                locations_text = ', '.join([f'Latitude: {loc["lat"]}, Longitude: {loc["lng"]} que corresponde a {addr}' for loc, addr in locations])
                result_message += f'Para resolver tal problema, o Mapper.AI recomenda a avaliação da viabilidade de {suggestion} nos seguintes locais: {locations_text}.\n'
        else:
            result_message += f'Para resolver tal problema, o Mapper.AI recomenda a avaliação da viabilidade de {suggestion}.\n'
    
    return result_message

# Chamando a função para exibir o resultado
start_address, end_address = get_congestion_addresses()
print_result(start_address, end_address, improvements)
