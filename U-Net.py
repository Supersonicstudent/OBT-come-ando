import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class UNet(nn.Module):
    def _init_(self):
        super(UNet, self)._init_()

        self.encoder1 = self.double_conv(3, 64)
        self.encoder2 = self.double_conv(64, 128)
        self.encoder3 = self.double_conv(128, 256)

        self.bottleneck = self.double_conv(256, 512)

        self.decoder3 = self.double_conv(512 + 256, 256)
        self.decoder2 = self.double_conv(256 + 128, 128)
        self.decoder1 = self.double_conv(128 + 64, 64)

        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))

        bottleneck = self.bottleneck(F.max_pool2d(enc3, 2))

        dec3 = self.decoder3(torch.cat([F.interpolate(bottleneck, scale_factor=2, mode='bilinear', align_corners=True), enc3], dim=1))
        dec2 = self.decoder2(torch.cat([F.interpolate(dec3, scale_factor=2, mode='bilinear', align_corners=True), enc2], dim=1))
        dec1 = self.decoder1(torch.cat([F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=True), enc1], dim=1))

        return torch.sigmoid(self.final_conv(dec1))

model = UNet()

#Preparando os dados

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import pandas as pd

class SegmentationDataset(Dataset):
    def _init_(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.traffic_data = pd.read_csv(traffic_data_path)

    def _len_(self):
        return len(self.image_files)

    def _getitem_(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('.jpg', '.png'))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        traffic_info = self.traffic_data[self.traffic_data['file'] == img_name].iloc[0]['traffic_value']

        return image, mask, traffic_info

# Definir transformações
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

#caminnhos
image_dir = 'caminho/para/imagens'
mask_dir = 'caminho/para/mascaras'
traffic_data_path = 'caminho/para/dados_de_trafego.csv'

#carregar dataset
dataset = SegmentationDataset(image_dir, mask_dir,traffic_data_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

#Treinando o modelo
import torch.optim as optim
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for images, masks, traffic_info in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        traffic_info = traffic_info.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader)}")

# Salvar o modelo treinado
torch.save(model.state_dict(), "caminho/para/salvar/modelo.pth")

#Usar o modelo para segmentação das ruas

# Carregar modelo treinado
model.load_state_dict(torch.load("caminho/para/salvar/modelo.pth"))
model.eval()

# Carregar e pré-processar imagem de teste
test_image_path = 'caminho/para/imagem_de_teste.jpg'
test_image = Image.open(test_image_path).convert("RGB")
test_image = transform(test_image).unsqueeze(0).to(device)

# Fazer previsão
with torch.no_grad():
    prediction = model(test_image)
    segmented_image = (prediction.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

# Exibir resultado
import matplotlib.pyplot as plt
plt.imshow(segmented_image, cmap='gray')
plt.show()

### A partir daqui é um modelo de associação de imagens e tráfego e a recomendação
### esse código provavelmente vai mudar

import networkx as nx

def create_road_network(segmented_images, traffic_data):
    G = nx.Graph()
    
    for idx, image in enumerate(segmented_images):
        traffic_value = traffic_data[idx]
        # Processar a imagem segmentada para identificar as ruas e adicionar nós/arestas ao grafo
        # Supondo que tenhamos uma função para isso: process_segmented_image(image)
        roads = process_segmented_image(image)
        
        for road in roads:
            start, end = road
            G.add_edge(start, end, weight=traffic_value)
    
    return G

def process_segmented_image(image):
    # Função fictícia que processa a imagem segmentada e retorna uma lista de ruas (start, end)
    # Esta função precisa ser implementada de acordo com o seu formato de dados
    return [(0, 1), (1, 2), (2, 3)]  # Exemplo fictício

def suggest_changes(G):
    # Algoritmo simples para sugerir mudanças
    # Por exemplo, encontrar os maiores gargalos e sugerir modificações
    # Supondo que tenhamos uma função para isso: find_bottlenecks(G)
    bottlenecks = find_bottlenecks(G)
    
    suggestions = []
    for bottleneck in bottlenecks:
        suggestions.append(f"Sugerir expansão da estrada entre {bottleneck[0]} e {bottleneck[1]}")
    
    return suggestions

def find_bottlenecks(G):
    # Função fictícia que encontra os maiores gargalos no grafo
    # Esta função precisa ser implementada de acordo com o seu formato de dados e necessidades
    return [(0, 1), (2, 3)]  # Exemplo fictício

# Supondo que temos as imagens segmentadas e os dados de tráfego
segmented_images = []  # Lista de imagens segmentadas
traffic_data = []  # Lista de dados de tráfego

# Criar a rede viária
G = create_road_network(segmented_images, traffic_data)

# Sugerir mudanças
suggestions = suggest_changes(G)
for suggestion in suggestions:
    print(suggestion)



### Visualisação de sugestões
import matplotlib.pyplot as plt

def visualize_suggestions(image, suggestions):
    plt.imshow(image)
    for suggestion in suggestions:
        # Visualizar a sugestão na imagem
        # Supondo que a sugestão contém coordenadas de pontos que precisam ser modificados
        start, end = suggestion  # Exemplo fictício
        plt.plot([start[0], end[0]], [start[1], end[1]], 'r')  # Linha vermelha para a sugestão
    
    plt.show()

# Supondo que temos uma imagem de teste e sugestões
test_image = Image.open('caminho/para/imagem_de_teste.jpg')
suggestions = [(0, 1), (2, 3)]  # Exemplo fictício
visualize_suggestions(test_image, suggestions)



### Modelo GAN de criação de imagens (Definição)
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Definir o gerador
class Generator(nn.Module):
    def _init_(self):
        super(Generator, self)._init_()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# Definir o discriminador
class Discriminator(nn.Module):
    def _init_(self):
        super(Discriminator, self)._init_()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# Inicializar modelos
netG = Generator().to(device)
netD = Discriminator().to(device)

###Treinamento do modelo
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        netD.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), 1., dtype=torch.float, device=device)

        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(batch_size, 100, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(0.)

        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        netG.zero_grad()
        label.fill_(1.)
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

    print(f'Epoch [{epoch}/{num_epochs}]  Loss_D: {errD.item():.4f}  Loss_G: {errG.item():.4f}  D(x): {D_x:.4f}  D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')

#Aplicando sugestões recomendações 
from PIL import Image, ImageDraw

def apply_suggestions(image_path, suggestions):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    
    for suggestion in suggestions:
        start, end = suggestion['start'], suggestion['end']
        draw.line([start, end], fill='red', width=5)
    
    return image

# Exemplo de uso
image_path = 'caminho/para/imagem_de_teste.jpg'
suggestions = [{'start': (50, 50), 'end': (150, 150)}, {'start': (100, 200), 'end': (300, 400)}]
modified_image = apply_suggestions(image_path, suggestions)
modified_image.show()
