
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
from dados_pre_processados import get_congestion_cords, identify_junctions, directions_result, get_congestion_addresses
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

# Função para exibir o resultado final
def return_result (start_address, end_address, improvements, coords):
    # Convertendo coordenadas para endereços
    for improvement in improvements:
       if improvement['location']:
         coords = improvement['location']
         addresses = convert_coords_to_addresses([coords])
         improvement['address'] = addresses[0]

    print(f'O engarrafamento na viagem especificada começa no trecho: {start_address} e termina em: {end_address}.')
    juncoes = improvement['location']
    if juncoes is None:
        print ("Para resolver tal problema, o Mapper.AI recomenda a avaliação da viabilidade de adicionar uma faixa na referida via")
    else:
        sugestao = improvement['suggestion']
        print( f'Para resolver tal problema, o Mapper.AI recomenda a avaliação da viabilidade de {sugestao} nos seguintes locais: {coords} que corresponde a {addresses}.')
# Chamando a função para exibir o resultado
start_address, end_address = get_congestion_addresses()
coords = []
return_result(start_address, end_address, improvements, coords)
def get_addresses():
    return start_address, end_address, improvements

start_address, end_address = get_congestion_addresses()
def get_addresses():
    return start_address, end_address