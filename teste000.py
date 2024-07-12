from flask import Flask, request
app = Flask(__name__)

@app.route('/process_data', methods=['POST'])
def process_data():

    return "O engarrafamento na viagem especificada começa no trecho: Epia, Em Frente A Novacap, Plano Piloto, Distrito Federal - Guará, Brasília - DF, Brazil e termina em: Smpw Trecho 2 Q 8 - Park Way Q 8 - Núcleo Bandeirante, Brasília - DF, Brazil. Para resolver tal problema, o Mapper.AI recomenda a avaliação da viabilidade de adicionar uma faixa na referida via"

if __name__ == '__main__':
    app.run(debug=True, host= '0.0.0.0', port=8000)