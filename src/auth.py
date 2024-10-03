import json

from flask import Flask, request, jsonify, current_app
from flask_restful import Resource, Api
import jwt
from functools import wraps


SECRET_KEY = 'bts-txt-ateez-skz-TUSPATRONES!'

def is_admin(current_user_id):
    try:
        # Obtener la referencia al documento del usuario
        firestore_db = current_app.db
        user_ref = firestore_db.collection('users').document(current_user_id)
        user = user_ref.get()

        if user.exists:
            user_data = user.to_dict()
            # Verificar si el usuario tiene el rol de 'admin'
            return user_data.get('role') == 'admin'
        else:
            print(f"User with ID {current_user_id} does not exist.")
            return False
    except Exception as e:
        print(f"Error checking admin status: {e}")
        return False


def token_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split()[1]

        if not token:
            return jsonify({'message': 'Token is missing!'}), 401

        try:
            data = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user_id = data['user_id']
            current_context = data.get('context', None)  # Extraer el contexto del token
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired!'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Token is invalid!'}), 401
        except Exception as e:
            return jsonify({'message': f'Token is invalid: {str(e)}'}), 401

        # Pasar `current_user_id` y `current_context` a la función original
        response = f(current_user_id=current_user_id, current_context=current_context, *args, **kwargs)

        # Solo se serializa si es un diccionario
        if isinstance(response, dict):
            return jsonify(response)
        return response

    return decorated_function