# src/auth.py
import json

from flask import Flask, request, jsonify, current_app
from flask_restful import Resource, Api
import jwt
from functools import wraps

SECRET_KEY = 'bts-txt-ateez-skz-TUSPATRONES!'


# Decorador para proteger las rutas
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split()[1]

        if not token:
            return jsonify({'message': 'Token is missing!'}), 401

        try:
            data = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user_id = data['user_id']
            current_context = data.get('context', None)  # Extraer el contexto del token
        except Exception as e:
            return jsonify({'message': f'Token is invalid: {str(e)}'}), 401

        return f(current_user_id=current_user_id, current_context=current_context, *args, **kwargs)

    return decorated