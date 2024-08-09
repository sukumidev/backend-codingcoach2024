import random
import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
from .utils import extract_years, extract_technologies, extract_level
from src.question_manager import get_question


lemmatizer = WordNetLemmatizer()


class Chatbot:
    def __init__(self, model, words, classes, intents):
        self.model = model
        self.words = words
        self.classes = classes
        self.intents = intents

    def predict_class(self, sentence):
        # Generar la bolsa de palabras para la oración de entrada
        p = self.bow(sentence)
        # Hacer la predicción usando el modelo
        res = self.model.predict(p)[0]
        # Determinar la clase con la mayor probabilidad
        return self.classes[np.argmax(res)]

    def clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def bow(self, sentence):
        # Tokenizar la oración y convertirla en un vector de "bag of words"
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(self.words)
        for s in sentence_words:
            for i, w in enumerate(self.words):
                if w == s:
                    bag[i] = 1
        return np.array([bag])

    def get_response(self, intent):
        for i in self.intents['intents']:
            if i['tag'] == intent:
                return random.choice(i['responses'])

    def chat(self, message):
        ints = self.predict_class(message)
        response = {"response": "", "next": ""}

        if ints == "start_technical_evaluation":
            # Iniciar la evaluación técnica
            self.current_context = "technical_evaluation"
            question = get_question(category="General Programming", difficulty="Medium", code_challenge=False)
            if question:
                response["response"] = question["question"]
                response["next"] = "Please provide your answer."
                # Aquí podrías guardar la pregunta seleccionada para comparar con la respuesta del usuario más tarde
                self.current_question = question
            else:
                response["response"] = "No suitable question found."

        elif self.current_context == "technical_evaluation":
            # Comparar la respuesta del usuario con la respuesta correcta
            if message.lower() == self.current_question["answer"].lower():
                response["response"] = "Correct! Well done."
            else:
                response["response"] = f"Incorrect. The correct answer is: {self.current_question['answer']}"

            # Seleccionar la siguiente pregunta
            next_question = get_question(category="General Programming", difficulty="Medium", code_challenge=False)
            if next_question:
                response["next"] = next_question["question"]
                self.current_question = next_question
            else:
                response["next"] = "No more questions available."

        return response
