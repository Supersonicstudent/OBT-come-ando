from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mydatabase.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Usuario(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nome = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    senha = db.Column(db.String(200), nullable=False)
    cidade = db.Column(db.String(80), nullable=False)

class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    descricao = db.Column(db.String(200), nullable=False)
    imagem_caminho = db.Column(db.String(200), nullable=False)
    usuario_id = db.Column(db.Integer, db.ForeignKey('usuario.id'), nullable=False)
    usuario = db.relationship('Usuario', backref=db.backref('posts', lazy=True))

class Comentario(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    conteudo = db.Column(db.String(500), nullable=False)
    post_id = db.Column(db.Integer, db.ForeignKey('post.id'), nullable=False)
    usuario_id = db.Column(db.Integer, db.ForeignKey('usuario.id'), nullable=False)
    post = db.relationship('Post', backref=db.backref('comentarios', lazy=True))
    usuario = db.relationship('Usuario', backref=db.backref('comentarios', lazy=True))

@app.route('/add_usuario', methods=['POST'])
def add_usuario():
    data = request.json
    hashed_password = generate_password_hash(data['senha'], method='pbkdf2:sha256')
    novo_usuario = Usuario(nome=data['nome'], email=data['email'], senha=hashed_password, cidade=data['cidade'])
    db.session.add(novo_usuario)
    db.session.commit()
    return jsonify({'message': 'Usuário adicionado com sucesso!'})

@app.route('/get_usuarios', methods=['GET'])
def get_usuarios():
    usuarios = Usuario.query.all()
    lista_usuarios = [{'id': u.id, 'nome': u.nome, 'email': u.email, 'cidade': u.cidade} for u in usuarios]
    return jsonify(lista_usuarios)

@app.route('/get_usuario/<int:id>', methods=['GET'])
def get_usuario(id):
    usuario = Usuario.query.get(id)
    if usuario is None:
        return jsonify({'message': 'Usuário não encontrado!'}), 404
    return jsonify({'id': usuario.id, 'nome': usuario.nome, 'email': usuario.email, 'cidade': usuario.cidade, 'senha': usuario.senha})

@app.route('/add_post', methods=['POST'])
def add_post():
    data = request.json
    novo_post = Post(descricao=data['descricao'], imagem_caminho=data['imagem_caminho'], usuario_id=data['usuario_id'])
    db.session.add(novo_post)
    db.session.commit()
    return jsonify({'message': 'Post adicionado com sucesso!'})

@app.route('/get_posts', methods=['GET'])
def get_posts():
    posts = Post.query.all()
    lista_posts = [{'id': p.id, 'descricao': p.descricao, 'imagem_caminho': p.imagem_caminho, 'usuario_id': p.usuario_id} for p in posts]
    return jsonify(lista_posts)

@app.route('/get_post/<int:id>', methods=['GET'])
def get_post(id):
    post = Post.query.get(id)
    if post is None:
        return jsonify({'message': 'Post não encontrado!'}), 404
    return jsonify({'id': post.id, 'descricao': post.descricao, 'imagem_caminho': post.imagem_caminho, 'usuario_id': post.usuario_id})

@app.route('/add_comentario', methods=['POST'])
def add_comentario():
    data = request.json
    novo_comentario = Comentario(conteudo=data['conteudo'], post_id=data['post_id'], usuario_id=data['usuario_id'])
    db.session.add(novo_comentario)
    db.session.commit()
    return jsonify({'message': 'Comentário adicionado com sucesso!'})

@app.route('/get_comentarios', methods=['GET'])
def get_comentarios():
    comentarios = Comentario.query.all()
    lista_comentarios = [{'id': c.id, 'conteudo': c.conteudo, 'post_id': c.post_id, 'usuario_id': c.usuario_id} for c in comentarios]
    return jsonify(lista_comentarios)

@app.route('/get_comentario/<int:id>', methods=['GET'])
def get_comentario(id):
    comentario = Comentario.query.get(id)
    if comentario is None:
        return jsonify({'message': 'Comentário não encontrado!'}), 404
    return jsonify({'id': comentario.id, 'conteudo': comentario.conteudo, 'post_id': comentario.post_id, 'usuario_id': comentario.usuario_id})

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    usuario = Usuario.query.filter_by(email=data['email']).first()
    if not usuario or not check_password_hash(usuario.senha, data['senha']):
        return jsonify({'message': 'Login falhou, verifique suas credenciais!'}), 401
    return jsonify({'message': 'Login bem-sucedido!', 'usuario': {'id': usuario.id, 'nome': usuario.nome, 'email': usuario.email, 'cidade': usuario.cidade}})

if __name__ == '__main__':
    with app.app_context():  # Adiciona o contexto da aplicação
        db.create_all()  # Cria as tabelas
    app.run(debug=True)
