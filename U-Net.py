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