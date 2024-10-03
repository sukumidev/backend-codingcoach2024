from flask import Flask
from flask_restful import Api
from .routes import initialize_routes
from firebase_admin import credentials, firestore
import firebase_admin

def create_app():
    app = Flask(__name__)

    app.config['SECRET_KEY'] = 'bts-txt-ateez-skz-TUSPATRONES!'

    # Inicializa Firebase si no está ya inicializado
    if not firebase_admin._apps:
        cred = credentials.Certificate('codingcoach-firebase-adminsdk-32vz3-d2307355dc.json')
        firebase_admin.initialize_app(cred)

    # Inicializa Firestore
    app.db = firestore.client()

    return app
