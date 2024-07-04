import random

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

#Configurando o sumo
import traci
import sumolib
import random
import pandas as pd

def create_sumo_simulation(net_file, route_file, simulation_time, lane_count, congestion_level):
    # Inicializar SUMO
    sumoBinary = sumolib.checkBinary('sumo')
    traci.start([sumoBinary, "-n", net_file, "-r", route_file])
    
    # Definir parâmetros de congestionamento
    if congestion_level == 'alto':
        additional_vehicles = 100
    elif congestion_level == 'médio':
        additional_vehicles = 50
    elif congestion_level == 'baixo':
        additional_vehicles = 20
    else:
        additional_vehicles = 0
    
    # Adicionar veículos adicionais para simular congestionamento
    for _ in range(additional_vehicles):
        route_id = random.choice(traci.route.getIDList())
        traci.vehicle.add(vehID=f'veh_{random.randint(1000, 9999)}', routeID=route_id, typeID='passenger')
    
    for step in range(simulation_time):
        traci.simulationStep()
    
    traci.close()
    
    # Calcular o índice de sucesso da implementação
    success_rate = calculate_success_rate()
    
    return success_rate

def calculate_success_rate():
    # Placeholder para calcular o índice de sucesso baseado na simulação
    # Pode ser ajustado conforme os critérios específicos do seu projeto
    success_rate = random.uniform(0, 10)
    return success_rate

# Exemplo de uso
net_file = 'your_network_file.net.xml'
route_file = 'your_route_file.rou.xml'
simulation_time = 3600  # 1 hora de simulação
lane_count = 3
congestion_level = 'alto'

success_rate = create_sumo_simulation(net_file, route_file, simulation_time, lane_count, congestion_level)
print(f"Índice de sucesso da implementação: {success_rate}")

# Adicionar ao DataFrame
traffic_data = pd.DataFrame(columns=['congestion_level', 'road_type', 'time_of_day', 'improvement_type', 'success_rate'])
traffic_data = traffic_data.append({
    'congestion_level': congestion_level,
    'road_type': f'{lane_count}_lanes',
    'time_of_day': 'peak_hour',
    'improvement_type': 'sample_implementation',
    'success_rate': success_rate
}, ignore_index=True)

print(traffic_data)

#Implementação e treinamento
import pandas as pd
import keras 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Executar simulações e coletar resultados
for implementation in implementations:
    success_rate = run_sumo_simulation(implementation)
    implementation['success_rate'] = success_rate

# Criar DataFrame a partir das implementações
df = pd.DataFrame(implementations)
df['success'] = df['success_rate'].apply(lambda x: 1 if x > 5 else 0)

# Preparar dados para o treinamento
X = df[['congestion_level', 'road_type', 'time_of_day', 'improvement_type']]
y = df['success']

# Codificar variáveis categóricas
label_encoders = {}
for column in ['road_type', 'time_of_day', 'improvement_type']:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# Padronizar os dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar a rede neural
model = Sequential()
model.add(Dense(32, input_dim=X.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=10, validation_data=(X_test, y_test))

# Avaliar o modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Model Accuracy: {accuracy*100:.2f}%')
