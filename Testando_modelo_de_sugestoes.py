from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def train_recommendation_model():
    # Utilizando o DataFrame fictício
    traffic_data = pd.DataFrame(data)
    
    # Conversão de variáveis categóricas para numéricas
    traffic_data = pd.get_dummies(traffic_data, columns=['road_type', 'time_of_day'])
    
    features = traffic_data.drop('improvement_type', axis=1)
    labels = traffic_data['improvement_type']
    
    model = RandomForestClassifier()
    model.fit(features, labels)
    
    return model

def suggest_improvements(congestion_coords, congestion_level, road_type, time_of_day, model):
    improvements = []
    
    for coord in congestion_coords:
        feature_vector = [congestion_level] + list(pd.get_dummies(pd.Series({'road_type': road_type, 'time_of_day': time_of_day})).iloc[0])
        improvement_suggestion = model.predict([feature_vector])
        
        improvements.append({
            'location': coord.tolist(),
            'suggestion': improvement_suggestion[0]
        })
    
    return improvements

# Exemplo de uso
recommendation_model = train_recommendation_model()
congestion_coords = np.argwhere(segmentada > 0.6)
improvements = suggest_improvements(congestion_coords, congestion_level=0.8, road_type='highway', time_of_day='morning', model=recommendation_model)

result = {
    'start_address': start_address,
    'end_address': end_address,
    'improvements': improvements
}

print(result)
