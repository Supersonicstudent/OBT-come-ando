from flask import Flask, request
from teste import get_congestion_addresses, identify_junctions, directions_result, get_congestion_cords, suggest_changes, convert_coords_to_addresses, print_result

app = Flask(__name__)

@app.route('/process_data', methods=['POST'])
def process_data():

    # Identificar junções
    start_location, end_location = get_congestion_cords()
    junctions = identify_junctions(directions_result, start_location, end_location)
    
    # Sugerir mudanças com base na segmentação
    improvements = suggest_changes(junctions)

    # Converter coordenadas para endereços
    for improvement in improvements:
        if improvement['location']:
            coords = improvement['location']
            addresses = convert_coords_to_addresses([coords])
            improvement['address'] = addresses[0]
    
    start_address, end_address = get_congestion_addresses()
    result_message = print_result(start_address, end_address, improvements)

    return result_message

if __name__ == '__main__':
    app.run(debug=True)

origem_geral = request.form.get('origem_geral')
destino_geral = request.form.get('destino_geral')