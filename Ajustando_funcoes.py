import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
import torchvision

# Dataset personalizado
class MapDataset(Dataset):
    def _init_(self, image_dir, mask_dir, traffic_data, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.traffic_data = traffic_data
        self.transform = transform
        self.images = os.listdir(image_dir)

    def _len_(self):
        return len(self.images)

    def _getitem_(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        traffic_info = self.traffic_data[idx]  # Assumindo que traffic_data está na mesma ordem

        return image, mask, traffic_info

# Transformações para aumento de dados
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor()
])

# Definindo o modelo U-Net
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5)  # Adiciona dropout
            )

        self.encoder1 = conv_block(3, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)
        self.encoder5 = conv_block(512, 1024)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = conv_block(128, 64)

        self.conv_last = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        enc5 = self.encoder5(self.pool(enc4))

        # Decoder
        dec4 = self.upconv4(enc5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        # Output
        out = self.conv_last(dec1)
        out = self.sigmoid(out)
        return out


# Diretórios das imagens e máscaras
image_dir = "path/to/images"
mask_dir = "path/to/masks"
traffic_data = np.load("path/to/traffic_data.npy")  # Carregando dados de tráfego

# Inicialização do dataset e dataloader
dataset = MapDataset(image_dir, mask_dir, traffic_data, transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Função de perda e otimizador
criterion = nn.BCELoss()
model = UNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Early stopping
patience = 5
best_loss = np.inf
trigger_times = 0

# Configurar o TensorBoard
writer = SummaryWriter('runs/unet_experiment')

# Parâmetros de treinamento
num_epochs = 20  # Número de épocas ajustável conforme necessidade

# Cross-validation
kf = KFold(n_splits=5)
fold_results = []

for fold, (train_index, val_index) in enumerate(kf.split(dataset)):
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_index)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_index)

    trainloader = DataLoader(dataset, batch_size=4, sampler=train_subsampler)
    valloader = DataLoader(dataset, batch_size=4, sampler=val_subsampler)

    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (images, masks, _) in enumerate(trainloader):  # Ignorando temporariamente os dados de tráfego
            images, masks = images.to(torch.float32), masks.to(torch.float32)
            outputs = model(images)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            # Registrar a perda de treinamento no TensorBoard
            writer.add_scalar('Loss/train', loss.item(), epoch * len(trainloader) + batch_idx)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (images, masks, _) in enumerate(valloader):  # Ignorando temporariamente os dados de tráfego
                images, masks = images.to(torch.float32), masks.to(torch.float32)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                # Registrar as imagens de entrada e saída no TensorBoard
                if batch_idx == 0:  # Apenas para o primeiro lote de cada época
                    img_grid = torchvision.utils.make_grid(images)
                    writer.add_image(f'images_fold_{fold}', img_grid, epoch)
                    pred_grid = torchvision.utils.make_grid(outputs)
                    writer.add_image(f'predictions_fold_{fold}', pred_grid, epoch)

        val_loss /= len(valloader)
        writer.add_scalar(f'Loss/val_fold_{fold}', val_loss, epoch)
        print(f'Epoch [{epoch+1}/{num_epochs}], Fold [{fold+1}/{kf.n_splits}], Validation Loss: {val_loss:.4f}')
        
        if val_loss < best_loss:
            best_loss = val_loss
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f'Early stopping on epoch {epoch+1}')
                break

    fold_results.append(val_loss)
    print(f'Fold {fold+1}, Validation Loss: {val_loss:.4f}')

print(f'Cross-Validation Loss: {np.mean(fold_results):.4f}')

# Fechar o TensorBoard writer
writer.close()

import torch
import numpy as np

# Função para processar a imagem segmentada e associar dados de tráfego
def process_segmented_image(segmented_image, traffic_data):
    # Extrair coordenadas das ruas
    roads = np.where(segmented_image == 1)
    road_coordinates = list(zip(roads[0], roads[1]))

    # Processar coordenadas para sugestões de melhorias
    suggestions = []
    threshold = 0.5  # Defina um limiar para decidir quando expandir as ruas
    for coord in road_coordinates:
        x, y = coord
        if 0 <= x < traffic_data.shape[0] and 0 <= y < traffic_data.shape[1]:  # Garantir que a coordenada esteja dentro dos limites
            traffic_value = traffic_data[x, y]
            suggestions.append((x, y, 'expand' if traffic_value > threshold else 'no change'))

    return suggestions

# Função para fazer a inferência do modelo e processar a imagem
def infer_and_process(model, data_loader, device):
    model.eval()  # Coloca o modelo em modo de avaliação
    with torch.no_grad():  # Desliga o cálculo do gradiente para a inferência
        for images, _, _ in data_loader:
            images = images.to(device)
            predictions = model(images)

            # Processar as previsões
            segmented_image = (predictions.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
            # Aqui você deve ter os dados de tráfego correspondentes às imagens
            traffic_data = np.random.rand(segmented_image.shape[0], segmented_image.shape[1])  # Exemplo de dados de tráfego
            suggestions = process_segmented_image(segmented_image, traffic_data)

            return suggestions  # Retorna as sugestões de melhorias

# Exemplo de uso da função infer_and_process
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Inicializando o modelo e movendo para o dispositivo
model = UNet().to(device)

# Inicializando o DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Exemplo de chamada da função
suggestions = infer_and_process(model, dataloader, device)
print("Sugestões de melhorias:", suggestions)
#------------------------------#

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

# Aqui vamos importar os dados pré-prpcessados dos endereços do engarrafamento, que estão no arquivo dados_pre_processados.py
#---------------------------------------------------------------------------------------------------------------------------#
from dados_pre_processados import get_congestion_addresses
def suggest_changes(image, model, start_address, end_address, segmentada):
    """
    Função para realizar a inferência no modelo U-Net e sugerir melhorias na infraestrutura.

    Parâmetros:
    - image: imagem do mapa de entrada
    - model: modelo U-Net treinado
    - start_address: endereço de início do engarrafamento
    - end_address: endereço de fim do engarrafamento
    - segmentada: imagem segmentada

    Retorna:
    - sugestões de melhorias na infraestrutura
    """
    model.eval()
    
    with torch.no_grad():
        input_image = torch.from_numpy(image).unsqueeze(0).float()
        prediction = model(input_image).squeeze().cpu().numpy()

    # Identificar regiões de engarrafamento na imagem segmentada
    congestion_area = segmentada

    # Processar a imagem segmentada para encontrar sugestões de melhorias
    improvements = []
    congestion_coords = np.argwhere(congestion_area > 0.6)
    
    if len(congestion_coords) > 0:
        for coord in congestion_coords:
            improvements.append({
                'location': coord.tolist(),
                'suggestion': 'Adicionar faixa extra'  # Sugestão de melhoria fictícia
            })

    # Montar o resultado final
    result = {
        'start_address': start_address,
        'end_address': end_address,
        'improvements': improvements
    }

    return result

# Exemplo de uso

# Obtendo os endereços de congestionamento
start_address, end_address = get_congestion_addresses()

# Supondo que 'segmentada' é a imagem segmentada
segmentada = np.array(Image.open('Segmentada.jpg').convert('L')) / 255.0

# Supondo que 'image' é a imagem segmentada pelo modelo U-Net
image = np.array(Image.open('Local do engarrafamento.jpg').convert('RGB')) / 255.0

result = suggest_changes(image, model, start_address, end_address, segmentada)

print(result)
# Supondo que temos as imagens segmentadas e os dados de tráfego
segmented_images = []  # Lista de imagens segmentadas
traffic_data = []  # Lista de dados de tráfego

# Criar a rede viária
G = create_road_network(segmented_images, traffic_data)

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
    def __init__(self):
        super(Generator, self).__init__()
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
    def __init__(self):
        super(Discriminator, self).__init__()
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

#NOTA: "device" não definido

#CERTAS INFORMAÇÕES SERÃO SUBSTITUIDAS PELOS DADOS