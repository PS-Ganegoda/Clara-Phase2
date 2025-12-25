import os
import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
lemmatizer = WordNetLemmatizer()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "app/data/intents.json")
ML_PATH = os.path.join(BASE_DIR, "app/ml")

os.makedirs(ML_PATH, exist_ok=True)

with open(DATA_PATH, encoding="utf-8") as file:
    intents = json.load(file)

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.wordpunct_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))

    if intent['tag'] not in classes:
        classes.append(intent['tag'])

words = sorted(set([lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]))
classes = sorted(set(classes))

pickle.dump(words, open(os.path.join(ML_PATH, "words.pkl"), "wb"))
pickle.dump(classes, open(os.path.join(ML_PATH, "classes.pkl"), "wb"))

# -----------------------
# Prepare training data
# -----------------------
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in document[0]]
    for word in words:
        bag.append(1 if word in word_patterns else 0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append(bag + output_row)

random.shuffle(training)
training = np.array(training)

trainX = training[:, :len(words)]
trainY = training[:, len(words):]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(trainY[0]), activation='softmax')
])

model.compile(loss="categorical_crossentropy",
              optimizer=tf.keras.optimizers.SGD(0.01, 0.9, nesterov=True),
              metrics=["accuracy"])

model.fit(trainX, trainY, epochs=200, batch_size=5, verbose=1)
model.save(os.path.join(ML_PATH, "chatbot_model.h5"))

print("✅ Training completed successfully")
