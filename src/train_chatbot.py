import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import pickle

# Inicializar lematizador
lemmatizer = WordNetLemmatizer()

# Cargar el archivo intents.json
with open('models/intents.json', 'r') as file:
    intents = json.load(file)

words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

# Procesar cada intención en intents.json
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenizar cada palabra en el patrón
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        # Añadir a nuestra lista de clases
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lematizar y bajar a minúsculas las palabras, ignorar ciertas palabras
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# Ordenar clases
classes = sorted(list(set(classes)))

# Guardar palabras y clases
pickle.dump(words, open('models/words.pkl', 'wb'))
pickle.dump(classes, open('models/classes.pkl', 'wb'))

# Preparar el entrenamiento
training = []
output_empty = [0] * len(classes)

# Crear el entrenamiento BoW para cada patrón
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # Salida es '1' para la clase actual
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Separar en train_x y train_y
train_x = np.array([t[0] for t in training])
train_y = np.array([t[1] for t in training])

# Verificar la forma de los datos antes de entrenar
print(f"train_x shape: {train_x.shape}")
print(f"train_y shape: {train_y.shape}")

# Construir el modelo
try:
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))

    # Compilar el modelo
    sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # Entrenar el modelo
    hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

    # Guardar el modelo entrenado
    model.save('models/chatbot_model.keras', hist)
    print("Modelo creado y guardado")
except Exception as e:
    print(f"Ocurrió un error durante el entrenamiento o guardado del modelo: {e}")
