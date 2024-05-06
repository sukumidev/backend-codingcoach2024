from flask_restful import Resource, reqparse
from flask import current_app as app
from .utils import encrypt_password, verify_password, is_email_unique, get_new_user_id
import base64


class Index(Resource):
    def get(self):
        return {'message': '¡Holaa mundo te amo jungkook!'}


class CreateUser(Resource):

    def post(self):
        db = app.db
        parser = reqparse.RequestParser()
        parser.add_argument('name', required=True, help="name cannot be blank")
        parser.add_argument('email', required=True, help="email cannot be blank")
        parser.add_argument('password', required=True, help="password cannot be blank")
        parser.add_argument('age', type=int, required=True, help="age cannot be blank")
        data = parser.parse_args()

        # Genera un nuevo user_id
        new_user_id = get_new_user_id()
        data['user_id'] = str(new_user_id)

        if not is_email_unique(data['email'], db):
            return {"success": False, "message": "Email already in use"}, 400

        data['password'] = encrypt_password(data['password'])

        try:
            doc_ref = db.collection('users').document(data['user_id'])
            doc_ref.set(data)
            return {"success": True, "message": "User created successfully"}, 200
        except Exception as e:
            return {"success": False, "message": str(e)}, 400


class Login(Resource):

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('email', required=True, help="email cannot be blank")
        parser.add_argument('password', required=True, help="Password cannot be blank")
        data = parser.parse_args()
        db = app.db
        # Busca el usuario en la base de datos por user_id
        user_ref = db.collection('users').document(data['email']).get()
        if user_ref.exists:
            user = user_ref.to_dict()
            # Verifica la contraseña
            if verify_password(data['password'], user['password']):
                return {"success": True, "message": "Login successful"}, 200
            else:
                return {"success": False, "message": "Invalid password"}, 401
        else:
            return {"success": False, "message": "User not found"}, 404


class ListUsers(Resource):
    def get(self):
        db = app.db
        users_ref = db.collection('users')
        docs = users_ref.stream()  # Obtiene todos los documentos en la colección de usuarios

        users_list = []
        for doc in docs:
            user_data = doc.to_dict()
            user_data['user_id'] = doc.id  # Añade el ID del documento a los datos del usuario
            for key, value in user_data.items():
                if isinstance(value, bytes):
                    user_data[key] = base64.b64encode(value).decode('utf-8')
            users_list.append(user_data)

        return {'users': users_list}, 200
