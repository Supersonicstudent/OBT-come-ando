from gmaps import maps
from pprint import pprint
import googlemaps 
import requests
from PIL import Image
from io import BytesIO

def obter_imagem_mapa(api_key, location='São Paulo, Brazil', zoom=15, tamanho='600x400'):
    url = 'https://maps.googleapis.com/maps/api/staticmap'
    params = {
        'center': location,
        'zoom': zoom,
        'size': tamanho,
        'key': api_key
    }
    response = requests.get(url, params=params)
    image = Image.open(BytesIO(response.content))
    return image

# Exemplo de utilização
api_key = 'AIzaSyDdTREWbb7NJRvkBjReLpRdgNIyqJeLcbM'
imagem_mapa = obter_imagem_mapa(api_key)
imagem_mapa.show()



