import os
import time
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torchvision.models.segmentation as segmentation
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Caminhos para imagens e máscaras
images_dir = ''
masks_dir = ''

# Definindo a transformação das imagens e máscaras com Data Augmentation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),  # Aumento de dados
    transforms.RandomRotation(10),      # Aumento de dados
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class SingleImageDataset(Dataset):
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

# Carregando o dataset
dataset = SingleImageDataset(images_dir, masks_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Definindo o modelo DeepLabV3, função de perda e otimizador
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = segmentation.deeplabv3_resnet101(pretrained=False, num_classes=21)  # Ajuste num_classes conforme necessário
model.to(device)
criterion = nn.CrossEntropyLoss()  # DeepLabV3 normalmente usa CrossEntropyLoss
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True)

# Função para treinamento contínuo integrada com seu código existente
def treinamento_continuo(model, dataloader, criterion, optimizer, device, num_epocas_total, pausa_intervalo=10):
    melhor_modelo = None
    melhor_loss = float('inf')
    no_improvement = 0
    
    for i in range(num_epocas_total // 5):  # Dividindo em etapas de 5 épocas
        print(f'Iniciando etapa {i + 1} de treinamento...')
        
        # Treinar o modelo por 5 épocas
        for epoch in range(5):
            model.train()
            epoch_loss = 0.0
            for images, masks in dataloader:
                images = images.to(device)
                masks = masks.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)['out']  # A saída do DeepLabV3 é um dicionário
                loss = criterion(outputs, masks)  # CrossEntropyLoss espera os rótulos como inteiros
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            epoch_loss /= len(dataloader)
            print(f'Epoch {epoch+1}/{5}, Loss: {epoch_loss:.4f}')
            
            # Scheduler step
            scheduler.step(epoch_loss)
            
            # Check for early stopping
            if epoch_loss < melhor_loss:
                melhor_loss = epoch_loss
                no_improvement = 0
                torch.save(model.state_dict(), 'checkpoint/deeplabv3_checkpoint.pth')  # Salvar melhor modelo
            else:
                no_improvement += 1
                if no_improvement >= 3:  # Patience de 3 épocas
                    print("Early stopping triggered")
                    break
            
            # Adicionar pausa após cada época
            print(f'Pausa de {pausa_intervalo} segundos para resfriamento...')
            time.sleep(pausa_intervalo)
        
        # Carregar o melhor modelo após cada etapa de treinamento
        checkpoint_path = 'checkpoint/deeplabv3_checkpoint.pth'
        model.load_state_dict(torch.load(checkpoint_path))
        model.to(device)
    
    # Após completar o treinamento contínuo, salvar o melhor modelo final
    torch.save(model.state_dict(), 'modelo_final.pth')
    print('Treinamento contínuo concluído. Modelo final salvo.')

# Exemplo de uso:
num_epocas_total = 20  # Total de épocas desejado
treinamento_continuo(model, dataloader, criterion, optimizer, device, num_epocas_total, pausa_intervalo=300)
