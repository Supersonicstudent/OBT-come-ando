from flask import Flask, request
app = Flask(__name__)

@app.route('/process_data', methods=['POST'])
def process_data():
    start_address, end_address, improvements = get_addresses()
    result_message = return_result(start_address, end_address, improvements)

    return result_message

if __name__ == '__main__':
    app.run(debug=True)

origem_geral = request.form.get('origem_geral')
destino_geral = request.form.get('destino_geral')