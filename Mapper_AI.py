import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import googlemaps
import torchvision.models.segmentation as segmentation
from dados_pre_processados import identify_junctions, directions_result

# Caminhos para os arquivos de entrada e saída
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
    gmaps = googlemaps.Client(key='YOUR_GOOGLE_MAPS_API_KEY')  # Substitua com sua chave de API do Google Maps
    addresses = []

    for coord in coords:
        lat, lng = coord['lat'], coord['lng']

        try:
            reverse_geocode_result = gmaps.reverse_geocode(latlng=(lat, lng), result_type='street_address')

            if reverse_geocode_result:
                address = reverse_geocode_result[0]['formatted_address']
            else:
                address = "Endereço não encontrado"

        except Exception as e:
            address = f"Erro ao buscar endereço: {str(e)}"

        addresses.append(address)

    return addresses
