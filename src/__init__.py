from flask import Flask
from flask_restful import Api
from .routes import initialize_routes
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

def create_app():
    app = Flask(__name__)
    api = Api(app)

    # Inicializa Firebase
    cred = credentials.Certificate(
        'C:/Users/rctva/PycharmProjects/CodingCoach/codingcoach-firebase-adminsdk-32vz3-9f11a60448.json')
    firebase_admin.initialize_app(cred)

    # Inicializa Firestore
    app.db = firestore.client()

    # Inicializa las rutas
    initialize_routes(api)

    return app
