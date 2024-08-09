from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import json
from tensorflow.keras.models import load_model
from src.chatbot import Chatbot  # Asegúrate de que la importación sea correcta
from src.resources import Login, CreateUser
from src import create_app
from flask_cors import CORS

# Configurar la aplicación Flask
app = create_app()
CORS(app)

# Cargar el modelo entrenado
model = load_model('src/models/chatbot_model.h5')

# Cargar las palabras y clases desde los archivos pickle
words = pickle.load(open('src/models/words.pkl', 'rb'))
classes = pickle.load(open('src/models/classes.pkl', 'rb'))

# Cargar el archivo intents.json
with open('src/models/intents.json', 'r') as file:
    intents = json.load(file)

# Crear una instancia del Chatbot con todos los argumentos necesarios
chatbot = Chatbot(model, words, classes, intents)

# Definir rutas y lógica del servidor
@app.route('/chat', methods=['POST'])
def chat():
    message = request.json.get('message')
    ints = chatbot.predict_class(message)
    response = chatbot.get_response(ints)
    return jsonify(response)

@app.route('/create_user', methods=['POST'])
def register_user():
    return CreateUser().post()

@app.route('/login', methods=['POST'])
def login_user():
    return Login().post()

if __name__ == '__main__':
    app.run(debug=True)
