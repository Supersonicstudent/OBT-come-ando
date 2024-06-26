import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import layers, models
import matplotlib.pyplot as plt
import cv2

#Carregar dados de tráfego
traffic_data = pd.read_csv ('caminho/para/dados/de/tráfego')

#Carregar imagem de mapas
import os
from glob import glob

img_path = ("indefinido")
image_paths = glob('caminhois/para/imagens_de_mapas/*.png') 

images = [cv2.imread(img_path) for img_path in image_paths]

#Normalizar dados de tráfego
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
traffic_data_normalized = scaler.fit_transform(traffic_data)

#Normalizar imagens (supondo 256x256 pixels)

img = ("imagem recebida.png")
images_reized = [cv2.resize(img, (256, 256)) for img in images]
images_normalized = [img / 255.0 for img in images_reized]

#Divião de dados
from sklearn.model_selection import train_test_split

##dividir dados de tráfego 
train_data, test_data = train_test_split(traffic_data_normalized, test_size=0.2, random_state=42)
##dividr as imagens (supondo que temos imagens correspondentes a cada linha de dados de tráfego)
train_images, test_images = train_test_split(images_normalized, test_size=0.2, random_state=42)

#Rede neural
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,(3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation="sigmoid") #Supondo que a saída é binária (congestionamento ou não)
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#Treinamento do modelo com dados de treinamento
##convertendo listas de imagens em array numpy
train_images_np = np.array(train_images)
test_images_np = np.array(test_images)

#Treinamento
history = model.fit(train_images_np, train_data, epochs=10, validation_data=(test_images_np, test_data))

#Avaliação e ajuste do modelo com os dados de teste conforme o necessário
test_loss, test_acc = model.evaluate(test_images_np, test_data, verbose=2)
print(f'Test accuracy: {test_acc}')

#Visualização dos Resultados (verificação de performance)
##prever com  o modelo
predictions = model.predict(test_images_np)

#Plotar algumas imagens com suas previsões
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(test_images_np[i])
    plt.title(f'Prediction: {predictions[i]}')
    plt.axis('off')
plt.show()