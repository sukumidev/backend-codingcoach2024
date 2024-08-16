# src/__init__.py

from flask import Flask
from flask_restful import Api
from .routes import initialize_routes
from firebase_admin import credentials, firestore
import firebase_admin

def create_app(chatbot):
    app = Flask(__name__)

    app.config['SECRET_KEY'] = 'bts-txt-ateez-skz-TUSPATRONES!'

    # Inicializa Firebase
    cred = credentials.Certificate('C:/Users/rctva/PycharmProjects/CodingCoach/codingcoach-firebase-adminsdk-32vz3-d2307355dc.json')
    firebase_admin.initialize_app(cred)

    # Inicializa Firestore
    app.db = firestore.client()

    return app