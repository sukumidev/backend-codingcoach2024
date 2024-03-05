from flask_restful import Resource, reqparse
from flask import current_app as app
from .utils import encrypt_password, verify_password



class Index(Resource):
    def get(self):
        return {'message': '¡Hola, mundoo0o0o0o0oo!'}


class CreateUser(Resource):

    def post(self):
        db = app.db
        parser = reqparse.RequestParser()
        parser.add_argument('user_id', required=True, help="user_id cannot be blank")
        parser.add_argument('name', required=True, help="name cannot be blank")
        parser.add_argument('email', required=True, help="email cannot be blank")
        parser.add_argument('password', required=True, help="password cannot be blank")
        parser.add_argument('age', type=int, required=True, help="age cannot be blank")
        data = parser.parse_args()

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
        parser.add_argument('user_id', required=True, help="user_id cannot be blank")
        parser.add_argument('password', required=True, help="Password cannot be blank")
        data = parser.parse_args()
        db = app.db
        # Busca el usuario en la base de datos por user_id
        user_ref = db.collection('users').document(data['user_id']).get()
        if user_ref.exists:
            user = user_ref.to_dict()
            # Verifica la contraseña
            if verify_password(data['password'], user['password']):
                return {"success": True, "message": "Login successful"}, 200
            else:
                return {"success": False, "message": "Invalid password"}, 401
        else:
            return {"success": False, "message": "User not found"}, 404
