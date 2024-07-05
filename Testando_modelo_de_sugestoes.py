import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Lista de implementações possíveis
improvement_types = ['lane_expansion', 'traffic_light_timing', 'road_closure', 'bus_lane']

# Função para gerar implementações aleatórias
def generate_random_implementations(n=10):
    implementations = []
    for _ in range(n):
        implementation = {
            'congestion_level': random.uniform(0, 1),
            'road_type': random.choice(['highway', 'city_street', 'rural_road']),
            'time_of_day': random.choice(['morning', 'afternoon', 'evening', 'night']),
            'improvement_type': random.choice(improvement_types)
        }
        implementations.append(implementation)
    return implementations

# Exemplo de implementações
implementations = generate_random_implementations()
print(implementations)

# Configurando o sumo
import traci

def simulate_in_sumo(improvement, congestion_level):
    sumo_cmd = [
        'sumo',
        '-c', r'C:\Users\steve\OneDrive\Área de Trabalho\OBT\OBT\OBT\sumoconfig.xml',
        '--additional-files', 'your_additional_file.add.xml'
    ]

    if congestion_level == 0:
        # Configurações para congestionamento baixo
        pass
    elif congestion_level == 1:
        # Configurações para congestionamento médio
        pass
    else:
        # Configurações para congestionamento alto
        pass

    traci.start(sumo_cmd)
    step = 0
    while step < 3600:
        traci.simulationStep()
        step += 1
    traci.close()

    success_rate = random.uniform(0, 1)
    return success_rate

# Implementação e treinamento
class TrafficImprovementModel(nn.Module):
    def __init__(self):
        super(TrafficImprovementModel, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

def train_model(model, dataloader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for data in dataloader:
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Calcula a média da perda durante a época
        epoch_loss /= len(dataloader)

        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')

def simulate_and_train(improvement_options, num_simulations=100):
    model = TrafficImprovementModel()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    all_data = []
    for _ in range(num_simulations):
        improvement = random.choice(improvement_options)
        congestion_level = random.choice([0, 1, 2])  # 0: baixo, 1: médio, 2: alto
        success_rate = simulate_in_sumo(improvement, congestion_level)
        
        input_vector = torch.tensor([improvement['road_width'], improvement['num_lanes'], improvement['speed_limit'], congestion_level], dtype=torch.float32)
        all_data.append((input_vector, torch.tensor(success_rate, dtype=torch.float32)))
    
    dataloader = DataLoader(all_data, batch_size=32, shuffle=True)
    train_model(model, dataloader, criterion, optimizer)

# Lista de melhorias disponíveis para a simulação
improvement_options = [
    {'road_width': 10, 'num_lanes': 2, 'speed_limit': 50},
    {'road_width': 15, 'num_lanes': 3, 'speed_limit': 60},
    # Outros tipos de melhorias
]
# Agora chame a função simulate_and_train para rodar as simulações e treinar a rede neural
simulate_and_train(improvement_options)
