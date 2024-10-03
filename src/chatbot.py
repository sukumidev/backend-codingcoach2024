import numpy as np
import random
import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer


class Chatbot:
    def __init__(self, model, words, classes, intents):
        self.model = model
        self.words = words
        self.classes = classes
        self.intents = intents
        self.lemmatizer = WordNetLemmatizer()

    def clean_up_sentence(self, sentence):
        print(f"Cleaning up sentence: {sentence}")
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        print(f"Tokenized and lemmatized words: {sentence_words}")
        return sentence_words

    def bow(self, sentence):
        print(f"Generating bag of words for sentence: {sentence}")
        sentence_words = self.clean_up_sentence(sentence)
        bow_vector = [0] * len(self.words)
        for word in sentence_words:
            for i, w in enumerate(self.words):
                if w == word:
                    bow_vector[i] = 1
        print(f"BoW Vector: {bow_vector}")
        return np.array(bow_vector)

    def predict_class(self, sentence):
        print(f"Predicting class for sentence: {sentence}")
        bow_vector = self.bow(sentence)
        res = self.model.predict(np.array([bow_vector]))[0]
        print(f"Prediction result: {res}")
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": self.classes[r[0]], "probability": str(r[1])})
        print(f"Predicted intents: {return_list}")
        return return_list

    def get_response(self, intent_list, intents_json):
        tag = intent_list[0]['intent']
        print(f"Retrieving response for intent: {tag}")
        intents = intents_json['intents']
        for i in intents:
            if i['tag'] == tag:
                response = random.choice(i['responses'])
                print(f"Selected response: {response}")
                return response

    def chat(self, msg, context=None, user_id=None):
        print(f"Chat function called with message: {msg}")
        intents = self.predict_class(msg)
        response_text = self.get_response(intents, self.intents)

        response = {
            "response": response_text,
            "next": ""
        }
        print(f"Final chatbot response: {response}")
        return response, None