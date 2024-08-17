# app.py
import json
import pickle

from flask import Flask, session
from flask_restful import Api
from keras.src.saving import load_model

from src import create_app
from src.chatbot import Chatbot
from src.routes import initialize_routes
from flask_cors import CORS
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
def test_server():
    print("Server is running...")

# Cargar el modelo, palabras, clases e intents
model = load_model('src/models/chatbot_model.h5')
words = pickle.load(open('src/models/words.pkl', 'rb'))
classes = pickle.load(open('src/models/classes.pkl', 'rb'))
with open('src/models/intents.json', 'r') as file:
    intents = json.load(file)

# Crear una instancia del chatbot
chatbot_instance = Chatbot(model, words, classes, intents)

# Crear la aplicación Flask y configurar CORS
app = create_app(chatbot_instance)  # Esta es la única vez que se crea `app`
CORS(app)
app.config['SECRET_KEY'] = 'bts-txt-ateez-skz-TUSPATRONES!'

# Inicializar la API con la instancia de `app`
api = Api(app)

# Inicializar las rutas con la instancia de `api` y `chatbot_instance`
initialize_routes(api, chatbot_instance)

# Agregar una ruta básica para probar
@app.route('/test', methods=['GET'])
def test_route():
    print("Test route hit")
    return {"message": "Test route is working!"}

if __name__ == '__main__':
    test_server()
    app.run(debug=True)
