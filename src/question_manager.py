import random
import pandas as pd

# Cargar el dataset de preguntas
questions_df = pd.read_csv('src/datasets/questions.csv')


def get_question(category=None, difficulty=None, code_challenge=False):
    # Filtrar por categoría, dificultad y si es un code challenge
    filtered_questions = questions_df.copy()

    if category:
        filtered_questions = filtered_questions[filtered_questions['Category'] == category]

    if difficulty:
        filtered_questions = filtered_questions[filtered_questions['Difficulty Points'] == difficulty]

    if code_challenge:
        filtered_questions = filtered_questions[filtered_questions['CodeChallenge'] == code_challenge]

    if not filtered_questions.empty:
        # Seleccionar una pregunta aleatoria
        question = filtered_questions.sample(n=1).iloc[0]
        return {
            "question": question["Question"],
            "answer": question["Answer"],
            "difficulty": question["Difficulty Points"],
            "code_challenge": question["CodeChallenge"]
        }
    else:
        return None