import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import imgaug.augmenters as iaa

# Definir o caminho para suas imagens e dados de tráfego
image_dir = 'caminho/para/imagens'
traffic_data = 'caminho/para/dados_de_trafego.csv'

# Função para carregar e pré-processar imagens
def load_and_preprocess_image(filepath, target_size=(416, 416)):
    image = Image.open(filepath).convert('RGB')
    image = image.resize(target_size)
    image = np.array(image) / 255.0  # Normalização
    return image

# Exemplo de carregamento de imagens
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]
images = np.array([load_and_preprocess_image(f) for f in image_files])

# Carregar dados de tráfego
traffic_df = pd.read_csv(traffic_data)

# Combinar dados de tráfego com imagens (exemplo simples)
# Assumimos que os dados de tráfego têm uma coluna 'file' que corresponde aos nomes dos arquivos de imagem
traffic_df['image'] = traffic_df['file'].apply(lambda x: load_and_preprocess_image(os.path.join(image_dir, x)))

# Converter dataframe para numpy array
traffic_images = np.array(list(traffic_df['image']))

# Aumento de dados (opcional) usando imgaug
augmenter = iaa.Sequential([
    iaa.Fliplr(0.5),  # Flip horizontalmente 50% das imagens
    iaa.Affine(rotate=(-20, 20)),  # Rotacionar imagens entre -20 e 20 graus
    iaa.ScaleX((0.8, 1.2)),  # Escalar largura das imagens entre 80% e 120%
    iaa.ScaleY((0.8, 1.2)),  # Escalar altura das imagens entre 80% e 120%
])

# Exemplo de aplicação de aumento de dados
augmented_images = augmenter(images=images)

# Agora, 'images' contém as imagens pré-processadas
# E 'augmented_images' contém as imagens aumentadas

##REDE NEURAL
