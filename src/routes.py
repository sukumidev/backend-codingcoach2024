# routes.py
import json

from flask_restful import Resource
from flask import Response, jsonify, request, session
from src.resources import Login, CreateUser, ListUsers
from src.auth import token_required
from src.chatbot import Chatbot


class ChatResource(Resource):
    @token_required
    def post(self, current_user_id=None, current_context=None):
        message = request.json.get('message')
        print(f"Received message: {message}")

        response, new_token = self.chatbot.chat(message, current_context, current_user_id)
        if not response['response']:
            response = {
                "response": "I didn't quite understand that. Could you please rephrase?",
                "next": ""
            }

        # Incluir el nuevo token en la respuesta
        return jsonify(response=response, token=new_token)


class ProtectedEndpoint(Resource):
    @token_required
    def get(self, current_user_id=None):
        print("Entering ProtectedEndpoint.get()")

        # Verificar que current_user_id es lo esperado
        print(f"Current user ID type: {type(current_user_id)}, value: {current_user_id}")

        try:
            # Preparar los datos de respuesta
            response_data = {
                "message": "This is a protected endpoint.",
                "user_id": current_user_id  # Aquí debería ser una cadena, no un objeto
            }
            print(f"Response data before serialization: {response_data}")

            # Convertir el diccionario a JSON y devolver la respuesta
            response_json = json.dumps(response_data)
            print(f"Serialized JSON response: {response_json}")

            return Response(response_json, status=200, mimetype='application/json')

        except Exception as e:
            print(f"An error occurred in ProtectedEndpoint: {str(e)}")
            return Response(json.dumps({"error": "An unexpected error occurred."}), status=500,
                            mimetype='application/json')


def initialize_routes(api, chatbot):
    api.add_resource(ChatResource, '/chat', resource_class_args=(chatbot,))
    api.add_resource(CreateUser, '/create_user')
    api.add_resource(Login, '/login')
    api.add_resource(ProtectedEndpoint, '/protected-endpoint')
    api.add_resource(ListUsers, '/users')
