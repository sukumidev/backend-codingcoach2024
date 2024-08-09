from flask import Flask
from flask_restful import Api
from .routes import initialize_routes
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

def create_app():
    app = Flask(__name__)
    api = Api(app)

    cred = credentials.Certificate('C:/Users/rctva/PycharmProjects/CodingCoach/codingcoach-firebase-adminsdk-32vz3-51054acc2d.json')
    firebase_admin.initialize_app(cred)

    app.db = firestore.client()

    initialize_routes(api)

    return app
