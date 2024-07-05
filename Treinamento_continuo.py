import os
import time
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torchvision.models.segmentation as segmentation

# Caminhos para imagens e máscaras
images_dir = r'C:\Users\steve\OneDrive\Área de Trabalho\OBT\OBT\OBT\Imagens'
masks_dir = r'C:\Users\steve\OneDrive\Área de Trabalho\OBT\OBT\OBT\Máscaras'

# Definindo a transformação das imagens e máscaras
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset personalizado para carregar imagens de satélite e máscaras de ruas
class SatelliteStreetDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_files = sorted(os.listdir(images_dir))
        self.mask_files = sorted(os.listdir(masks_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # L = modo de imagem em escala de cinza
        
        if self.transform:
            image = self.transform(image)
            mask = transforms.Resize((256, 256))(mask)  # Redimensionar máscara para o tamanho adequado
            mask = transforms.ToTensor()(mask)  # Convertendo para tensor
            mask = mask.squeeze(0).long()  # Remover o canal e converter para rótulos inteiros

        return image, mask

# Carregando o dataset utilizando DatasetFolder do PyTorch
dataset = SatelliteStreetDataset(images_dir, masks_dir, transform=transform)
batch_size = 2  # Ajuste conforme necessário
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Definindo o modelo DeepLabV3, função de perda e otimizador
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Atualizando o modelo para usar 'weights' em vez de 'pretrained'
model = segmentation.deeplabv3_resnet50(weights=None).to(device)

# Função de perda e otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Agendador de taxa de aprendizado
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# Função para salvar o estado do modelo
def save_checkpoint(model, optimizer, scheduler, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, path)

# Função para carregar o estado do modelo
def load_checkpoint(model, optimizer, scheduler, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch']

# Função para treinar o modelo com pausas para resfriamento
def treinamento_continuo(model, dataloader, criterion, optimizer, scheduler, device, num_epocas_total, pausa_intervalo=200):
    for epoca in range(num_epocas_total):
        model.train()
        running_loss = 0.0
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        epoch_loss = running_loss / len(dataloader)
        print(f'Epoch {epoca + 1}/{num_epocas_total}, Loss: {epoch_loss:.4f}')
        print(f'Pausa de {pausa_intervalo} segundos para resfriamento...')
        time.sleep(pausa_intervalo)
        
        # Salvar checkpoint após cada época
        checkpoint_path = f'checkpoint_epoch_{epoca + 1}.pth'
        save_checkpoint(model, optimizer, scheduler, epoca + 1, checkpoint_path)

# Inicializando o treinamento contínuo
num_epocas_total = 5
checkpoint_path = 'checkpoint_final.pth'
if os.path.exists(checkpoint_path):
    start_epoch = load_checkpoint(model, optimizer, scheduler, checkpoint_path)
else:
    start_epoch = 0

treinamento_continuo(model, dataloader, criterion, optimizer, scheduler, device, num_epocas_total, pausa_intervalo=200)
