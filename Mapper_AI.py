import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.models.segmentation as segmentation
import numpy as np
import warnings
from dados_pre_processados import identify_junctions, directions_result, get_congestion_addresses
import googlemaps

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

# Função para sugerir mudanças com base na segmentação
def suggest_changes(segmented_path):
    segmentada = Image.open(segmented_path).convert('L')
    segmentada = np.array(segmentada)
    congestion_coords = np.argwhere(segmentada > 0.6)
    improvements = []
    
    # Identificar junções com base no resultado das direções
    junctions = identify_junctions(directions_result)
    
    if junctions:
        for junction in junctions:
            improvements.append({
                'location': {'lat': junction['lat'], 'lng': junction['lng']},
                'suggestion': 'adicionar um semáforo'
            })
        if len(congestion_coords) <= 2:
            improvements.append({
                'location': None,
                'suggestion': 'adicionar uma faixa extra'
            })
    elif len(congestion_coords) > 0:
        for coord in congestion_coords:
            improvements.append({
                'location': {'lat': coord[0], 'lng': coord[1]},
                'suggestion': 'adicionar uma faixa extra'
            })

    return improvements

# Função para converter coordenadas em endereços usando Google Maps Reverse Geocoding
def convert_coords_to_addresses(coords):
    gmaps = googlemaps.Client(key='AIzaSyDdTREWbb7NJRvkBjReLpRdgNIyqJeLcbM')  # Substitua com sua chave de API do Google Maps
    addresses = []
    
    for coord in coords:
        lat, lng = coord['lat'], coord['lng']
        
        try:
            reverse_geocode_result = gmaps.reverse_geocode(latlng=(lat, lng), result_type='street_address')
            
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
start_address, end_address = get_congestion_addresses()
junctions = identify_junctions(directions_result)

improvements = suggest_changes(output_segmented_path)

# Convertendo coordenadas para endereços
for improvement in improvements:
    if improvement['location']:
        coords = improvement['location']
        addresses = convert_coords_to_addresses([coords])
        improvement['address'] = addresses[0]

# Função para exibir o resultado final
def print_result(start_address, end_address, improvements):
    print(f'O engarrafamento na viagem especificada começa no trecho: {start_address} e termina em: {end_address}.')
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
                    print(f'Para resolver tal problema, o Mapper.AI recomenda a avaliação da viabilidade de {suggestion}.')
                else:
                    loc = locations[0][0]
                    addr = locations[0][1]
                    print(f'Para resolver tal problema, o Mapper.AI recomenda a avaliação da viabilidade de {suggestion} no seguinte local: Latitude: {loc["lat"]}, Longitude: {loc["lng"]} que corresponde a {addr}.')
            else:
                locations_text = ', '.join([f'Latitude: {loc["lat"]}, Longitude: {loc["lng"]} que corresponde a {addr}' for loc, addr in locations])
                print(f'Para resolver tal problema, o Mapper.AI recomenda a avaliação da viabilidade de {suggestion} nos seguintes locais: {locations_text}.')
        else:
            print(f'Para resolver tal problema, o Mapper.AI recomenda a avaliação da viabilidade de {suggestion}.')

# Chamando a função para exibir o resultado
print_result(start_address, end_address, improvements)