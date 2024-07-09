from flask import Flask, request, jsonify
import os
from Mapper_AI import segment_image, suggest_changes, convert_coords_to_addresses, model, output_segmented_path
from dados_pre_processados import get_congestion_addresses

app = Flask(__name__)

@app.route('/process_image', methods=['POST'])
def process_image():
    origem_geral = request.form.get('origem_geral')
    destino_geral = request.form.get('destino_geral')

    file = request.files['image']
    image_path = os.path.join('/tmp', file.filename)
    file.save(image_path)
    
    # Processar a imagem de entrada e realizar a segmentação
    segment_image(image_path, model, output_segmented_path)
    
    # Obter os endereços de congestão e o resultado das direções
    start_address, end_address, directions_result = get_congestion_addresses(origem_geral, destino_geral)
    
    # Sugerir mudanças com base na segmentação
    improvements = suggest_changes(output_segmented_path)

    # Converter coordenadas para endereços
    for improvement in improvements:
        if improvement['location']:
            coords = improvement['location']
            addresses = convert_coords_to_addresses([coords])
            improvement['address'] = addresses[0]

    result = {
        'start_address': start_address,
        'end_address': end_address,
        'improvements': improvements
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
