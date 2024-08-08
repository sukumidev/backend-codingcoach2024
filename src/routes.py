from .resources import CreateUser, Login, Index

def initialize_routes(api):
    api.add_resource(CreateUser, '/create_user')
    api.add_resource(Login, '/login')
    api.add_resource(Index, '/')