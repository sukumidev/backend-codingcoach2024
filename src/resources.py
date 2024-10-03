import base64
import os
import random
import subprocess
import traceback
import datetime
import pytz
import bcrypt
import nltk
import numpy as np
import pandas as pd
from flask_restful import reqparse
from nltk.stem import WordNetLemmatizer
import math
from google.cloud.firestore_v1._helpers import DatetimeWithNanoseconds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import current_app as app, send_file, make_response, current_app, Response
from flask_restful import Resource
from flask import request, jsonify, session
import pickle
import json
from .auth import is_admin, token_required
import jwt
from firebase_admin import firestore

from .chatbot_demo import obtener_pregunta_de_firebase, evaluar_respuesta, generar_retroalimentacion, \
    generar_pdf_preguntas

SECRET_KEY = 'bts-txt-ateez-skz-TUSPATRONES!'

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
        parser.add_argument('role', required=False, help="Role of the user (e.g., admin, user). Default is 'user'")

        data = parser.parse_args()
        db = app.db

        # Hash la contraseña antes de almacenarla
        hashed_password = bcrypt.hashpw(data['password'].encode('utf-8'), bcrypt.gensalt())

        # Configura el rol del usuario, si no se proporciona, se establece como 'user'
        user_role = data['role'] if data['role'] else 'user'

        user_data = {
            'name': data['name'],
            'email': data['email'],
            'password': base64.b64encode(hashed_password).decode('utf-8'),  # Almacenar como cadena base64
            'age': data['age'],
            'role': user_role
        }

        db.collection('users').document(data['email']).set(user_data)
        return {"success": True, "message": "User created successfully"}, 201


class Login(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('email', required=True, help="Email cannot be blank")
        parser.add_argument('password', required=True, help="Password cannot be blank")
        data = parser.parse_args()
        db = app.db  # Asegúrate de que este `app` es el mismo que inicializaste en `create_app`

        try:
            users_ref = db.collection('users')
            query = users_ref.where('email', '==', data['email']).stream()

            user = None
            for doc in query:
                user = doc.to_dict()
                user['id'] = doc.id  # Asegúrate de que el ID de usuario se incluya en la respuesta
                break

            if user:
                stored_password = base64.b64decode(user['password'].encode('utf-8'))
                if bcrypt.checkpw(data['password'].encode('utf-8'), stored_password):
                    # Generar el token JWT
                    token = jwt.encode({
                        'user_id': user['email'],
                        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)
                    }, SECRET_KEY, algorithm="HS256")

                    return {
                        "success": True,
                        "message": "Login successful",
                        "token": token,  # Incluir el token en la respuesta
                        "user": user
                    }, 200
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
    def __init__(self, chatbot):
        self.chatbot = chatbot

    def post(self):
        print("Incoming POST request to /chat")
        message = request.json.get('message')
        print(f"Received message: {message}")
        response, _ = self.chatbot.chat(message)
        print(f"Chatbot response: {response}")

        # Asegurarse de que la respuesta sea un diccionario serializable
        try:
            return jsonify(response)
        except Exception as e:
            print(f"Error while serializing response: {str(e)}")
            return jsonify({"error": "Failed to process response"}), 500


class RandomQuestion(Resource):
    def get(self):
        # Obtener una pregunta aleatoria del dataset
        random_question = questions_df.sample(n=1).iloc[0]
        print(random_question)  # Para depuración

        # Convertir los valores de int64 a un tipo serializable
        return jsonify({
            "question": random_question['Question'],
            "answer": random_question['Answer'],
            "category": random_question['Category'],
            "difficulty": int(random_question['Difficulty Points'])  # Convertir a int
        })


class FilteredRandomQuestion(Resource):
    def get(self):
        category = request.args.get('category')
        difficulty = request.args.get('difficulty')

        # Filtrar por categoría o dificultad si se proporcionan
        filtered_df = questions_df
        if category:
            filtered_df = filtered_df[filtered_df['Category'] == category]
        if difficulty:
            filtered_df = filtered_df[filtered_df['Difficulty Points'] == int(difficulty)]  # Convertir dificultad a int

        if filtered_df.empty:
            return jsonify({"error": "No questions found for the specified filters"}), 404

        # Obtener una pregunta aleatoria del dataset filtrado
        random_question = filtered_df.sample(n=1).iloc[0]

        # Convertir los valores de int64 a un tipo serializable
        return jsonify({
            "question": random_question['Question'],
            "answer": random_question['Answer'],
            "category": random_question['Category'],
            "difficulty": int(random_question['Difficulty Points'])
        })


class Logout(Resource):
    @token_required
    def post(self, current_user_id=None, current_context=None):
        try:
            # Invalida la sesión del usuario (si estás usando sesiones)
            session.pop('user_id', None)
            session.pop('user_name', None)
            session.pop('current_context', None)
            session.clear()

            # Devuelve una respuesta JSON simple
            return {"message": "Successfully logged out"}, 200

        except Exception as e:
            # Manejar errores y asegurarse de que se retorne un JSON válido
            return {"error": str(e)}, 500


class AddQuestion(Resource):
    @token_required
    def post(self, current_user_id=None, current_context=None):
        if not is_admin(current_user_id):
            return {"message": "Unauthorized"}, 403

        parser = reqparse.RequestParser()
        parser.add_argument('question', required=True, help="Question cannot be blank")
        parser.add_argument('answer', required=True, help="Answer cannot be blank")
        parser.add_argument('category', required=True, help="Category cannot be blank")
        parser.add_argument('difficulty', required=True, help="Difficulty cannot be blank")
        parser.add_argument('challenge', required=False)
        parser.add_argument('starter_code', required=False)
        parser.add_argument('expected_output', required=False)

        data = parser.parse_args()
        db = app.db

        question_data = {
            'question': data['question'],
            'answer': data['answer'],
            'category': data['category'],
            'difficulty': data['difficulty'],
            'challenge': data.get('challenge', False),
            'starter_code': data.get('starter_code', ''),
            'expected_output': data.get('expected_output', ''),
            'created_by': current_user_id,
            'created_at': firestore.SERVER_TIMESTAMP
        }

        try:
            db.collection('questions').add(question_data)
            return {"success": True, "message": "Question added successfully"}, 201
        except Exception as e:
            return {"success": False, "message": f"An error occurred: {str(e)}"}, 500


class UpdateQuestion(Resource):
    @token_required
    def put(self, id, current_user_id=None, current_context=None):
        # Verificar si el usuario es administrador
        if not is_admin(current_user_id):
            return {"message": "Unauthorized"}, 403

        parser = reqparse.RequestParser()
        parser.add_argument('question', required=False)
        parser.add_argument('answer', required=False)
        parser.add_argument('category', required=False)
        parser.add_argument('difficulty', required=False)
        parser.add_argument('challenge', required=False)
        parser.add_argument('starter_code', required=False)
        parser.add_argument('expected_output', required=False)
        data = parser.parse_args()

        db = app.db

        # Obtener la referencia del documento a actualizar
        question_ref = db.collection('questions').document(id)
        question = question_ref.get()

        if not question.exists:
            return {"message": "Question not found"}, 404

        # Actualizar los campos
        updated_data = {key: value for key, value in data.items() if value is not None}

        # Realizar la actualización en la base de datos
        question_ref.update(updated_data)

        return {"success": True, "message": "Question updated successfully"}, 200


class DeleteQuestion(Resource):
    @token_required
    def delete(self, id, current_user_id=None, current_context=None):
        if not is_admin(current_user_id):
            return {"message": "Unauthorized"}, 403

        db = firestore.client()

        try:
            # Buscar el documento por su ID
            question_ref = db.collection('questions').document(str(id))
            question = question_ref.get()

            if question.exists:
                # Eliminar el documento
                question_ref.delete()
                return {"message": "Question deleted successfully"}, 200
            else:
                return {"message": "Question not found"}, 404
        except Exception as e:
            return {"message": f"An error occurred: {str(e)}"}, 500


class GetAllQuestions(Resource):
    def get(self):
        db = firestore.client()
        questions_ref = db.collection('questions')
        questions = questions_ref.stream()

        all_questions = []
        for question in questions:
            question_data = question.to_dict()
            question_data['id'] = question.id  # Añadir el ID del documento a los datos
            all_questions.append(question_data)

        return jsonify(all_questions)


class GetUserProfile(Resource):
    @token_required
    def get(self, current_user_id=None, current_context=None):
        db = app.db
        user_ref = db.collection('users').document(current_user_id)
        user_doc = user_ref.get()

        if user_doc.exists:
            return jsonify(user_doc.to_dict())
        else:
            return {"message": "User not found"}, 404


class UpdateUserProfile(Resource):
    @token_required
    def put(self, current_user_id=None, current_context=None):
        parser = reqparse.RequestParser()
        parser.add_argument('name', required=False)
        parser.add_argument('age', required=False)
        data = parser.parse_args()

        db = app.db
        user_ref = db.collection('users').document(current_user_id)
        user_doc = user_ref.get()

        if user_doc.exists:
            # Actualizar la información del usuario
            updated_data = {}
            if data['name']:
                updated_data['name'] = data['name']
            if data['age']:
                updated_data['age'] = data['age']

            if updated_data:
                user_ref.update(updated_data)
                return {"message": "User profile updated successfully"}, 200
            else:
                return {"message": "No data provided to update"}, 400
        else:
            return {"message": "User not found"}, 404


class DeleteUserProfile(Resource):
    @token_required
    def delete(self, current_user_id=None, current_context=None):
        db = app.db
        user_ref = db.collection('users').document(current_user_id)
        user_doc = user_ref.get()

        if user_doc.exists:
            user_ref.delete()
            return {"message": "User account deleted successfully"}, 200
        else:
            return {"message": "User not found"}, 404


class SubmitAnswer(Resource):
    @token_required
    def post(self, current_user_id=None, current_context=None):
        data = request.get_json()
        db = app.db

        question_id = data.get('question_id')
        is_correct = data.get('is_correct')
        response_time = data.get('response_time')
        score = data.get('score')

        # Validar que todos los campos necesarios están presentes
        if not all([question_id, is_correct is not None, response_time, score]):
            return {"message": "All fields are required."}, 400

        # Crear un nuevo documento en la colección "user_responses"
        new_response_ref = db.collection('user_responses').document()
        new_response_ref.set({
            'user_id': current_user_id,
            'question_id': question_id,
            'answered_at': datetime.datetime.utcnow(),
            'is_correct': is_correct,
            'response_time': response_time,
            'score': score
        })

        response_data = {
            "success": True,
            "message": "Answer recorded successfully"
        }

        return response_data, 201  # Devolver el diccionario directamente


class GetUserResponses(Resource):
    @token_required
    def get(self, current_user_id=None, current_context=None):
        db = app.db

        # Obtener todas las respuestas del usuario
        responses_ref = db.collection('user_responses').where('user_id', '==', current_user_id).stream()

        responses = []
        for response in responses_ref:
            responses.append(response.to_dict())

        if not responses:
            return {"message": "No responses found for this user."}, 404

        return {"responses": responses}, 200


db = firestore.client()


class InterviewResource(Resource):
    @token_required
    def post(self, current_user_id=None, current_context=None):
        print("Solicitud recibida")
        try:
            # Obtener datos de la solicitud
            data = request.get_json()
            print(f"Datos recibidos: {data}")  # Para debugging

            # Si el usuario está respondiendo una pregunta
            if 'answer' in data and 'question_id' in data:
                question_id = data['question_id']
                answer = data['answer']

                # Obtener la entrevista más reciente del usuario
                interview_query = db.collection('user_responses').document(current_user_id).collection(
                    'interviews').order_by("timestamp", direction=firestore.Query.DESCENDING).limit(1).get()

                if not interview_query:
                    return {"error": "No interview found for this user"}, 404

                interview_ref = interview_query[0].reference
                interview_data = interview_query[0].to_dict()

                # Obtener la lista de preguntas realizadas
                preguntas_realizadas = interview_data.get('questions', [])
                if question_id >= len(preguntas_realizadas):
                    return {"error": "Invalid question ID"}, 400

                # Si la respuesta es "salir" o "terminar", finalizar la entrevista
                if answer.lower() in ["salir", "terminar", "exit", "quit"]:
                    return self.finalizar_entrevista(interview_ref, interview_data)

                # Evaluar la respuesta del usuario
                respuesta_correcta = preguntas_realizadas[question_id]['Answer']
                points = preguntas_realizadas[question_id]['Difficulty_Points']
                keywords = preguntas_realizadas[question_id].get('Keywords', '')
                puntaje = evaluar_respuesta(answer, respuesta_correcta, keywords, points)

                # Generar la retroalimentación usando GPT-3.5
                retroalimentacion = generar_retroalimentacion(
                    preguntas_realizadas[question_id]['Question'], answer, interview_data['language'])

                # Actualizar la entrevista con la respuesta, retroalimentación y puntaje por pregunta
                interview_snapshot = interview_ref.get()
                interview_data = interview_snapshot.to_dict()

                # Si no existe el campo 'question_scores', inicializarlo como una lista vacía
                current_scores = interview_data.get('question_scores', [])

                # Añadir el nuevo puntaje
                current_scores.append(puntaje)

                # Actualizar la entrevista con la respuesta, retroalimentación y puntaje por pregunta
                interview_ref.update({
                    'answers': firestore.ArrayUnion([answer]),
                    'feedback': firestore.ArrayUnion([retroalimentacion]),
                    'score': firestore.Increment(puntaje),  # Asegura que el puntaje total se incrementa correctamente
                    'current_question_index': firestore.Increment(1),
                    'question_scores': current_scores  # Actualizar manualmente la lista de puntajes
                })

                # Verificar si el current_question_index ha alcanzado el límite (10 preguntas)
                if len(interview_data.get('answers', [])) >= 10:
                    return self.finalizar_entrevista(interview_ref, interview_data)

                # Verificar si hay más preguntas
                siguiente_pregunta_id = question_id + 1
                if siguiente_pregunta_id < len(preguntas_realizadas):
                    siguiente_pregunta = preguntas_realizadas[siguiente_pregunta_id]['Question']
                    return {
                        "question": siguiente_pregunta,
                        "question_id": siguiente_pregunta_id,
                        "feedback": retroalimentacion,
                        "score": puntaje
                    }, 200
                else:
                    # No hay más preguntas, finalizar la entrevista
                    return self.finalizar_entrevista(interview_ref, interview_data)

            # Si es la primera solicitud, comenzamos la entrevista
            elif 'language' in data and 'technologies' in data and 'experience' in data:
                idioma = data.get('language', 'Spanish')
                experiencia = data.get('experience', 0)
                tecnologias = data.get('technologies', [])

                if not tecnologias:
                    return {"error": "No technologies provided"}, 400

                print(f"Iniciando entrevista en {idioma} con tecnologías: {tecnologias} y experiencia: {experiencia}")

                # Obtener las preguntas iniciales
                preguntas_realizadas = []
                for tecnologia in tecnologias:
                    preguntas = obtener_pregunta_de_firebase(tecnologia, idioma)
                    if preguntas:
                        preguntas_realizadas.extend(preguntas)

                if not preguntas_realizadas:
                    return {"error": "No questions available for the selected technologies and language"}, 400

                random.shuffle(preguntas_realizadas)
                current_time = datetime.datetime.now(pytz.utc)  # Asegúrate de usar la clase 'datetime'
                timestamp_id = current_time.strftime('%Y%m%d%H%M%S')

                # Guardar el estado de la entrevista en Firestore
                interview_ref = db.collection('user_responses').document(current_user_id).collection(
                    'interviews').document(timestamp_id)
                interview_ref.set({
                    'user_id': current_user_id,
                    'questions': preguntas_realizadas,
                    'current_question_index': 0,
                    'answers': ["Esperando primera respuesta..."],
                    'feedback': ["Feedback pendiente..."],
                    'score': 0,
                    'question_scores': [],  # Lista vacía para almacenar puntajes por pregunta
                    'level': "",
                    'language': idioma,
                    'experience': experiencia,
                    'timestamp': firestore.SERVER_TIMESTAMP  # Guarda también el timestamp como campo
                })

                # Devolver la primera pregunta
                return {"question": preguntas_realizadas[0]['Question'], "question_id": 0}, 200

        except Exception as e:
            print(f"Error: {str(e)}")  # Para debugging
            return {"error": str(e), "trace": traceback.format_exc()}, 500

    def finalizar_entrevista(self, interview_ref, interview_data):
        # Calcular el nivel del usuario según el puntaje acumulado
        puntajes_preguntas = interview_data.get('question_scores', [])
        puntaje_total = interview_data.get('score', 0)  # Usar el puntaje total almacenado
        num_preguntas_respondidas = interview_data.get('current_question_index', 0)  # Número de preguntas respondidas

        # Asegurarse de que no se divida por cero si no hay preguntas respondidas
        promedio_puntaje = puntaje_total / num_preguntas_respondidas if num_preguntas_respondidas > 0 else 0
        promedio_puntaje = round(promedio_puntaje, 2)

        if promedio_puntaje <= 30:
            nivel = "Junior"
        elif promedio_puntaje <= 60:
            nivel = "Intermediate"
        else:
            nivel = "Senior"

        # Actualizar la entrevista con el nivel final y el puntaje promedio como el nuevo "score"
        interview_ref.update({
            'level': nivel,
            'completed': True,
            'score': promedio_puntaje  # Guardar el promedio en el campo "score"
        })

        return {
            "message": "Entrevista finalizada",
            "total_score": promedio_puntaje,  # Mostrar el promedio calculado
            "level": nivel  # Nivel según el puntaje promedio
        }, 200


def convert_interview_data(interview):

    interview_data = interview.to_dict()

    # Iterar sobre los campos para convertir tipos no serializables
    for key, value in interview_data.items():
        if isinstance(value, DatetimeWithNanoseconds):
            interview_data[key] = value.isoformat()  # Convertir DatetimeWithNanoseconds a string

    return interview_data


class UserInterviewsResource(Resource):
    def get(self):
        user_id = request.args.get('user_id')

        if not user_id:
            return {"error": "User ID is required"}, 400

        try:
            # Obtener las entrevistas de un usuario desde la subcolección 'interviews'
            user_ref = db.collection('user_responses').document(user_id).collection('interviews')
            interviews = user_ref.stream()

            interviews_list = []
            for interview in interviews:
                interview_data = convert_interview_data(interview)
                interview_data['id'] = interview.id  # Añadir el ID del documento
                interviews_list.append(interview_data)

            # Serialización manual usando json.dumps
            return Response(json.dumps(interviews_list), status=200, mimetype='application/json')

        except Exception as e:
            return {"error": str(e)}, 500


class PreferredLanguagesResource(Resource):
    def get(self, user_id):
        try:
            # Obtener las entrevistas del usuario desde Firebase
            interviews_ref = db.collection('user_responses').document(user_id).collection('interviews')
            interviews = interviews_ref.stream()

            language_count = {}
            total_questions = 0

            # Contar los lenguajes/categorías usados en las entrevistas
            for interview in interviews:
                interview_data = interview.to_dict()
                for question in interview_data.get('questions', []):
                    # Verificar si la pregunta tiene el campo "Category"
                    if isinstance(question, dict) and 'Category' in question:
                        category = question['Category']  # Aquí usamos 'Category' con C mayúscula
                        if category in language_count:
                            language_count[category] += 1
                        else:
                            language_count[category] = 1
                        total_questions += 1  # Contar todas las preguntas para el cálculo del porcentaje

            # Calcular el porcentaje de uso de cada lenguaje
            preferred_languages = [
                {
                    'language': lang,
                    'count': count,
                    'percentage': (count / total_questions) * 100
                }
                for lang, count in language_count.items()
            ]

            return preferred_languages, 200

        except Exception as e:
            return {"error": str(e)}, 500


class ScoreProgressResource(Resource):
    def get(self, user_id):
        try:
            # Obtener las entrevistas del usuario desde Firebase
            interviews_ref = db.collection('user_responses').document(user_id).collection('interviews')
            interviews = interviews_ref.stream()

            scores = []
            total_score = 0
            total_interviews = 0

            # Recoger todos los puntajes de las entrevistas
            for interview in interviews:
                interview_data = interview.to_dict()
                score = interview_data.get('score', 0)
                scores.append(score)
                total_score += score
                total_interviews += 1

            # Calcular el puntaje promedio
            average_score = total_score / total_interviews if total_interviews > 0 else 0

            return {
                "scores": scores,
                "average_score": average_score
            }, 200

        except Exception as e:
            return {"error": str(e)}, 500


class ScoreByTechnologyResource(Resource):
    def get(self, user_id):
        try:
            # Obtener las entrevistas del usuario desde Firebase
            interviews_ref = db.collection('user_responses').document(user_id).collection('interviews')
            interviews = interviews_ref.stream()

            technology_scores = {}
            technology_counts = {}

            # Recorrer las entrevistas para obtener los puntajes por tecnología
            for interview in interviews:
                interview_data = interview.to_dict()
                answers = interview_data.get('answers', [])
                question_scores = interview_data.get('question_scores', [])

                # Recorrer las preguntas y sus respuestas
                for idx, question in enumerate(interview_data.get('questions', [])):
                    # Verificar si la pregunta tiene el campo "Category"
                    if isinstance(question, dict) and 'Category' in question:
                        category = question['Category']

                        # Obtener el puntaje correspondiente de las respuestas del usuario
                        score_obtenido = question_scores[idx] if idx < len(question_scores) else 0

                        # Sumar el puntaje obtenido y contar cuántas preguntas hay por cada tecnología
                        if category in technology_scores:
                            technology_scores[category] += score_obtenido
                            technology_counts[category] += 1
                        else:
                            technology_scores[category] = score_obtenido
                            technology_counts[category] = 1

            # Calcular el promedio de puntaje por tecnología
            technology_averages = [
                {
                    'technology': tech,
                    'average_score': round(technology_scores[tech] / technology_counts[tech], 2)
                }
                for tech in technology_scores
            ]

            return technology_averages, 200

        except Exception as e:
            return {"error": str(e)}, 500


class InterviewSummaryResource(Resource):
    def get(self, user_id):
        try:
            # Obtener las entrevistas del usuario desde Firebase
            interviews_ref = db.collection('user_responses').document(user_id).collection('interviews')
            interviews = interviews_ref.stream()

            interview_summaries = []

            for interview in interviews:
                interview_data = interview.to_dict()
                interview_id = interview.id  # Obtenemos el ID de la entrevista

                # Usar un set para eliminar tecnologías duplicadas
                technologies = set(
                    question.get('Category', 'Desconocido') for question in interview_data.get('questions', []))
                total_score = interview_data.get('score', 0)

                # Convertir el timestamp si existe
                timestamp = interview_data.get('timestamp')
                if isinstance(timestamp, DatetimeWithNanoseconds):
                    timestamp = timestamp.isoformat()

                interview_summaries.append({
                    'interview_id': interview_id,
                    'technologies': list(technologies),  # Convertir set a lista para JSON
                    'total_score': total_score,
                    'timestamp': timestamp  # Añadir la fecha de la entrevista
                })

            # Ordenar las entrevistas por fecha (timestamp) de la más reciente a la más antigua
            sorted_interviews = sorted(interview_summaries, key=lambda x: x['timestamp'], reverse=True)

            return sorted_interviews, 200

        except Exception as e:
            return {"error": str(e)}, 500


class InterviewDetailResource(Resource):
    def get(self, user_id, interview_id):
        try:
            # Obtener la entrevista específica del usuario por ID
            interview_ref = db.collection('user_responses').document(user_id).collection('interviews').document(
                interview_id)
            interview = interview_ref.get()

            if interview.exists:
                interview_data = convert_interview_data(interview)

                # Obtener el índice de la última pregunta respondida
                current_question_index = interview_data.get('current_question_index', 0)
                answers = interview_data.get('answers', [])

                # Filtrar solo las preguntas respondidas y verificar los índices
                interview_data['questions'] = [
                    {**q, "UserAnswer": answers[i]} for i, q in enumerate(interview_data.get('questions', []))
                    if i < current_question_index and i < len(answers)  # Asegúrate de que el índice esté en el rango
                ]

                # Eliminar tecnologías duplicadas
                technologies = set(q.get('Category', 'Desconocido') for q in interview_data['questions'])
                interview_data['technologies'] = list(technologies)

                # Limpiar NaN y serializar los datos
                interview_data_clean = self.replace_nan(interview_data)

                # Serialización manual usando json.dumps
                return Response(json.dumps(interview_data_clean), status=200, mimetype='application/json')
            else:
                return {"error": "Entrevista no encontrada"}, 404

        except Exception as e:
            return {"error": str(e)}, 500

    @staticmethod
    def replace_nan(data):
        """ Reemplaza cualquier NaN en los datos con None. """
        if isinstance(data, dict):
            return {key: InterviewDetailResource.replace_nan(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [InterviewDetailResource.replace_nan(item) for item in data]
        elif isinstance(data, float) and math.isnan(data):
            return None
        else:
            return data


class UserClassificationResource(Resource):
    def __init__(self):
        self.perfiles_tecnologias = {
            "Full-stack Developer": ["JavaScript", "Node.js", "React", "PHP", "SQL", "Linux"],
            "Frontend Developer": ["JavaScript", "React"],
            "Backend Developer": ["Node.js", "Python", "Java", "C", "C++", "C#", "PHP", "SQL", "Go", "Linux"],
            "Data Engineer": ["Python", "SQL", "Java", "Linux"],
            "DevOps Engineer": ["Linux", "Python", "Go", "Docker", "Kubernetes"],
            "Data Scientist": ["Python", "SQL", "Linux"],
            "Mobile Developer": ["Java", "C#", "React Native"],
            "Embedded Systems Engineer": ["C", "C++", "Linux"],
            "Systems Engineer": ["C", "C++", "Java", "Python", "Go", "Linux"],
            "Game Developer": ["C++", "C#", "Java"],
            "Cybersecurity Specialist": ["Linux", "Python", "SQL"],
            "Cloud Engineer": ["Python", "Linux", "Go"]
        }

    # Función para clasificar al usuario y normalizar los porcentajes
    def clasificar_usuario_proporcional_normalizado(self, tecnologias_usuario):
        coincidencias = {}

        for perfil, tecnologias_perfil in self.perfiles_tecnologias.items():
            count_coincidencias = len([tech for tech in tecnologias_usuario if tech in tecnologias_perfil])
            total_tecnologias_perfil = len(tecnologias_perfil)

            porcentaje = (count_coincidencias / total_tecnologias_perfil) * 100 if total_tecnologias_perfil > 0 else 0
            coincidencias[perfil] = porcentaje

        suma_total = sum(coincidencias.values())
        if suma_total > 0:
            coincidencias_normalizadas = {perfil: (porcentaje / suma_total) * 100 for perfil, porcentaje in coincidencias.items()}
        else:
            coincidencias_normalizadas = coincidencias

        return coincidencias_normalizadas

    # Función para guardar o actualizar respuestas en Firebase
    def actualizar_respuestas_firebase(self, user_id, coincidencias_normalizadas):
        user_ref = db.collection('user_responses').document(user_id)
        profiles_ref = user_ref.collection('profiles')

        for perfil, nuevo_porcentaje in coincidencias_normalizadas.items():
            doc = profiles_ref.document(perfil).get()
            if doc.exists:
                datos_existentes = doc.to_dict()
                porcentaje_existente = datos_existentes.get('porcentaje', 0)
                promedio_porcentaje = (porcentaje_existente + nuevo_porcentaje) / 2
            else:
                promedio_porcentaje = nuevo_porcentaje

            profiles_ref.document(perfil).set({
                'perfil': perfil,
                'porcentaje': promedio_porcentaje
            })

    # Método POST para manejar la clasificación y guardar los datos
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('user_id', required=True, help="user_id es requerido")
        parser.add_argument('tecnologias', type=list, location='json', required=True, help="Lista de tecnologías es requerida")
        args = parser.parse_args()

        user_id = args['user_id']
        tecnologias_usuario = args['tecnologias']

        # Clasificación de las tecnologías del usuario
        coincidencias_normalizadas = self.clasificar_usuario_proporcional_normalizado(tecnologias_usuario)

        # Actualizar o crear el registro en Firebase
        self.actualizar_respuestas_firebase(user_id, coincidencias_normalizadas)

        return coincidencias_normalizadas, 200


class ProfilesPieChartResource(Resource):
    def get(self, user_id):
        try:
            user_ref = db.collection('user_responses').document(user_id)
            profiles_ref = user_ref.collection('profiles')

            profiles_docs = profiles_ref.stream()
            profiles_data = []

            for doc in profiles_docs:
                profile_data = doc.to_dict()
                # Redondear el porcentaje a 2 decimales sin convertirlo en cadena
                porcentaje_redondeado = round(profile_data.get("porcentaje", 0), 2)
                profiles_data.append({
                    "profile": profile_data.get("perfil"),
                    "percentage": porcentaje_redondeado  # Devolver como número
                })

            if not profiles_data:
                return {"message": "No se encontraron datos de perfiles para este usuario."}, 404

            return profiles_data, 200

        except Exception as e:
            return {"error": str(e)}, 500
