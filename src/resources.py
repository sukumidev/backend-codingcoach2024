import base64
import os
import random
import subprocess
import bcrypt
import nltk
import numpy as np
import pandas as pd
from flask_restful import reqparse
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import current_app as app
from flask_restful import Resource
from flask import request, jsonify
from tensorflow.keras.models import load_model
import pickle
import json
from src.chatbot import Chatbot

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

base_dir = os.path.abspath(os.path.dirname(__file__))

questions_df = pd.read_csv(os.path.join(base_dir, './datasets/questions.csv'))

questions = questions_df.to_dict(orient='records')

model = pickle.load(open('src/models/chatbot_model.pkl', 'rb'))

# Cargar el modelo y los intents
model = pickle.load(open('src/models/chatbot_model.pkl', 'rb'))

# Cargar los intents con json
with open('src/models/intents.json', 'r') as file:
    intents = json.load(file)


words = pickle.load(open(os.path.join(base_dir, './models/words.pkl'), 'rb'))
classes = pickle.load(open(os.path.join(base_dir, './models/classes.pkl'), 'rb'))
model = pickle.load(open(os.path.join(base_dir, './models/chatbot_model.pkl'), 'rb'))

user_data = {
    "state": "initial",
    "current_question": None,
    "score": 0
}


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict([p])[0]
    return classes[np.argmax(res)]


def get_response(intents, intent):
    for i in intents['intents']:
        if i['tag'] == intent:
            return random.choice(i['responses'])


def extract_years(message):
    years = [int(s) for s in message.split() if s.isdigit()]
    return years[0] if years else 0


def extract_technologies(message):
    return message.replace("and", ",").split(",")


def extract_level(message):
    if "junior" in message.lower():
        return "Junior"
    elif "intermediate" in message.lower():
        return "Intermediate"
    elif "senior" in message.lower():
        return "Senior"
    else:
        return "Unknown"


def save_user_profile(user_data):
    db = app.db
    profile_ref = db.collection('profiles').document()
    profile_ref.set({
        'profile': user_data["profile"],
        'experience_years': user_data["experience_years"],
        'technologies': user_data["technologies"],
        'level': user_data["level"]
    })


def classify_user():
    profile = user_data["profile"]
    experience_years = user_data["experience_years"]
    technologies = user_data["technologies"]
    level = user_data["level"]

    classification = {
        "profile": profile,
        "experience_years": experience_years,
        "technologies": technologies,
        "level": level
    }

    save_user_profile(user_data)

    return classification


def calculate_similarity(answer, correct_answer):
    vectorizer = TfidfVectorizer().fit_transform([answer, correct_answer])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0, 1]


class Index(Resource):
    def get(self):
        return {'message': 'yeonjun, soobin y hueningkaaaaaaaaaaaaai'}


class CreateUser(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('name', required=True, help="Name cannot be blank")
        parser.add_argument('email', required=True, help="Email cannot be blank")
        parser.add_argument('password', required=True, help="Password cannot be blank")
        parser.add_argument('age', required=True, help="Age cannot be blank")
        data = parser.parse_args()
        db = app.db

        # Hash la contraseña antes de almacenarla
        hashed_password = bcrypt.hashpw(data['password'].encode('utf-8'), bcrypt.gensalt())

        user_data = {
            'name': data['name'],
            'email': data['email'],
            'password': base64.b64encode(hashed_password).decode('utf-8'),  # Almacenar como cadena base64
            'age': data['age']
        }

        db.collection('users').document(data['email']).set(user_data)
        return {"success": True, "message": "User created successfully"}, 201


class Login(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('email', required=True, help="Email cannot be blank")
        parser.add_argument('password', required=True, help="Password cannot be blank")
        data = parser.parse_args()
        db = app.db

        try:
            users_ref = db.collection('users')
            query = users_ref.where('email', '==', data['email']).stream()

            user = None
            for doc in query:
                user = doc.to_dict()
                user['id'] = doc.id  # Ensure user ID is included in response
                break

            if user:
                stored_password = base64.b64decode(user['password'].encode('utf-8'))
                if bcrypt.checkpw(data['password'].encode('utf-8'), stored_password):
                    return {"success": True, "message": "Login successful", "user": user}, 200
                else:
                    return {"success": False, "message": "Invalid password"}, 401
            else:
                return {"success": False, "message": "User not found"}, 404
        except Exception as e:
            app.logger.error(f"Login error: {str(e)}")
            return {"success": False, "message": "An error occurred during login"}, 500


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


def get_next_question(user_data, questions_csv='questions.csv'):
    questions_df = pd.read_csv(questions_csv)

    # Filtrar preguntas según las tecnologías del usuario
    relevant_questions = questions_df[questions_df['Category'].isin(user_data['technologies'])]

    # Filtrar preguntas según los Difficulty Points y el nivel inicial del usuario
    difficulty_range = {
        "junior": (0, 10),
        "mid": (11, 24),
        "senior": (25, 100)
    }
    min_difficulty, max_difficulty = difficulty_range[user_data['level']]

    filtered_questions = relevant_questions[
        (relevant_questions['Difficulty Points'] >= min_difficulty) &
        (relevant_questions['Difficulty Points'] <= max_difficulty)
        ]

    # Seleccionar una pregunta al azar
    if len(filtered_questions) == 0:
        return None  # No hay preguntas que coincidan

    next_question = filtered_questions.sample(1).iloc[0]

    # Actualizar el puntaje total y el número de preguntas
    user_data['total_score'] += next_question['Difficulty Points']
    user_data['questions_so_far'] += 1

    # Calcular la dificultad para la siguiente pregunta basada en el promedio actual
    user_data['next_question_difficulty'] = user_data['total_score'] / user_data['questions_so_far']

    return next_question


class CompileCode(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('code', required=True, help="Code cannot be blank")
        parser.add_argument('language', required=True, help="Language cannot be blank")
        parser.add_argument('expected_outputs', required=True, type=dict, action='append',
                            help="Expected outputs cannot be blank")
        data = parser.parse_args()

        code = data['code']
        language = data['language']
        expected_outputs = data['expected_outputs']
        response = {"results": []}

        try:
            for test_case in expected_outputs:
                input_value = test_case['input']
                expected_output = test_case['output']

                if language == "javascript":
                    code_with_input = f"{code}\nconst result = isEvenOrOdd({input_value});\nconsole.log(result);"
                    process = subprocess.Popen(['node', '-e', code_with_input], stdin=subprocess.PIPE,
                                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                else:
                    response['results'].append(
                        {"input": input_value, "output": "Unsupported language", "correct": False})
                    continue

                output, error = process.communicate()
                if process.returncode == 0:
                    actual_output = output.decode('utf-8').strip()
                    is_correct = actual_output == str(expected_output)
                    response['results'].append(
                        {"input": input_value, "output": actual_output, "expected_output": str(expected_output),
                         "correct": is_correct})
                else:
                    response['results'].append({"input": input_value, "output": error.decode('utf-8').strip(),
                                                "expected_output": str(expected_output), "correct": False})

        except Exception as e:
            response['error'] = str(e)

        return jsonify(response)


class ChatResource(Resource):
    def __init__(self):
        # Cargar el modelo y los datos necesarios para inicializar Chatbot
        model = load_model('src/models/chatbot_model.h5')
        words = pickle.load(open('src/models/words.pkl', 'rb'))
        classes = pickle.load(open('src/models/classes.pkl', 'rb'))
        with open('src/models/intents.json', 'r') as file:
            intents = json.load(file)

        # Crear la instancia de Chatbot con todos los argumentos necesarios
        self.chatbot = Chatbot(model, words, classes, intents)

    def post(self):
        message = request.json.get('message')
        print(f"Received message: {message}")

        # Manejo explícito de mensajes iniciales
        if message.lower() in ["start", "hello", "hi"]:
            response = {
                "response": "Welcome! Let's start the interview. Can you tell me how many years of work experience you have?",
                "next": ""
            }
        else:
            response = self.chatbot.chat(message)
            if not response['response']:
                # Respuesta de seguridad en caso de que algo salga mal
                response = {
                    "response": "I didn't quite understand that. Could you please rephrase?",
                    "next": ""
                }

        print(f"Sending response: {response}")
        return jsonify(response)