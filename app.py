import json
import os
import pickle

from flask_restful import Api
from src.routes import initialize_routes
from src import create_app
from src.chatbot import Chatbot
from flask_cors import CORS


def test_server():
    print("Server is running...")

# Crear la aplicación Flask y configurar CORS
app = create_app()  # Esta es la única vez que se crea `app`
CORS(app)
app.config['SECRET_KEY'] = 'bts-txt-ateez-skz-TUSPATRONES!'

# Inicializar la API con la instancia de `app`
api = Api(app)

initialize_routes(api)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    test_server()
    app.run(debug=True, host='0.0.0.0', port=port)

