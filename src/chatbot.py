import random
import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
from .utils import extract_years, extract_technologies, extract_level
from src.question_manager import get_question
from flask import session,current_app
import jwt

lemmatizer = WordNetLemmatizer()


class Chatbot:
    def __init__(self, model, words, classes, intents):
        self.model = model
        self.words = words
        self.classes = classes
        self.intents = intents

    def generate_new_token(self, current_user_id, context):
        token = jwt.encode({
            'user_id': current_user_id,
            'context': context
        }, current_app.config['SECRET_KEY'], algorithm="HS256")
        return token

    def chat(self, message, current_context, current_user_id):
        response = {"response": "", "next": ""}

        print(f"Current context before: {current_context}")

        if current_context is None:
            current_context = "experience_question"
            response["response"] = "Welcome! Let's start the interview. Can you tell me how many years of work experience you have?"

        elif current_context == "experience_question":
            print(f"Received experience: {message}")
            current_context = "technology_question"
            response["response"] = "Thank you! Can you tell me which technologies you have experience with?"

        elif current_context == "technology_question":
            print(f"Received technologies: {message}")
            current_context = "technical_evaluation"
            response["response"] = "Great! Let's start the technical evaluation. Please answer the following question."
            question = get_question(category="General Programming", difficulty="Medium", code_challenge=False)
            if question:
                response["next"] = question["question"]
            else:
                response["next"] = "No suitable question found."

        elif current_context == "technical_evaluation":
            print(f"Evaluating answer: {message}")
            if message.lower() == "correct answer":  # Ajusta esta lógica según tu implementación
                response["response"] = "Correct! Well done."
            else:
                response["response"] = "Incorrect. The correct answer is: correct answer"

            # Aquí puedes actualizar el contexto o finalizar el proceso

        # Generar un nuevo token con el contexto actualizado
        new_token = self.generate_new_token(current_user_id, current_context)

        print(f"New context after: {current_context}")

        return response, new_token