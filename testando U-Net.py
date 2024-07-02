import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Dataset para uma única imagem
class SingleImageDataset(Dataset):
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform
        self.image = Image.open(image_path).convert("RGB")

    def __len__(self):
        return 1  # Apenas uma imagem

    def __getitem__(self, idx):
        image = self.image
        if self.transform:
            image = self.transform(image)
        return image

# Transformações
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Ajuste conforme necessário
    transforms.ToTensor()
])

# Caminho da imagem a ser testada
test_image_path = r"c:\Users\João Vitor\Pictures\Roblox\Saved Pictures\maparua1.jpg.jpg"  # Atualize este caminho

# Verifique se o arquivo existe
if not os.path.exists(test_image_path):
    print(f"Erro: Arquivo não encontrado em {test_image_path}")
else:
    print(f"Arquivo encontrado em {test_image_path}")

    # Inicialização do dataset e DataLoader
    dataset = SingleImageDataset(test_image_path, transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Definindo o modelo U-Net (supondo que o modelo já esteja definido e treinado)
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
                    nn.Dropout(0.5)
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
            enc1 = self.encoder1(x)
            enc2 = self.encoder2(self.pool(enc1))
            enc3 = self.encoder3(self.pool(enc2))
            enc4 = self.encoder4(self.pool(enc3))
            enc5 = self.encoder5(self.pool(enc4))
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
            out = self.conv_last(dec1)
            out = self.sigmoid(out)
            return out

    # Função para realizar a inferência
    def test_model(model, dataloader, device):
        model.eval()
        with torch.no_grad():
            for images in dataloader:
                images = images.to(device)
                predictions = model(images)
                segmented_images = (predictions.squeeze().cpu().numpy() > 0.6).astype(np.uint8)
                return images.squeeze().cpu().numpy(), predictions.squeeze().cpu().numpy(), segmented_images

    # Configuração do dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Inicializar o modelo
    model = UNet().to(device)

    # Carregar pesos treinados, se disponíveis
    # model.load_state_dict(torch.load('path/to/model_weights.pth'))

    # Teste o modelo
    original_images, predictions, segmented_images = test_model(model, dataloader, device)

    # Exibir a imagem original, a previsão do modelo e a imagem segmentada
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(np.transpose(original_images, (1, 2, 0)))
    axs[0].set_title("Imagem Original")

    axs[1].imshow(predictions, cmap='gray')
    axs[1].set_title("Previsão do Modelo")

    axs[2].imshow(segmented_images, cmap='gray')
    axs[2].set_title("Imagem Segmentada")

    for ax in axs:
        ax.axis('off')

    plt.show()


    #inteligencia artificial para modificar mapas e sugerir melhoria mna infraestrutura das ruas
    #testar o modelo u net usando uma imagem .jpg para indentificar as ruas na iagem
    #configurar tensoboard
    #codigo que envolve a inicialização do nodelo u net e a inferencia, com a função infer_and_process que processa as previsões e sugere melhorias na infraestrutura
