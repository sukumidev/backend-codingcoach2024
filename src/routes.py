from .resources import ChatResource, CreateUser, Login, Index, ListUsers, Chatbot, CompileCode

def initialize_routes(api):
    api.add_resource(CreateUser, '/create_user')
    api.add_resource(Login, '/login')
    api.add_resource(Index, '/')
    api.add_resource(ListUsers, '/users')
    api.add_resource(ChatResource, '/chat')
    api.add_resource(CompileCode, '/compile')
