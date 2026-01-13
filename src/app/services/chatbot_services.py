try:
    # This works in the Cloud (Northflank)
    import tflite_runtime.interpreter as tflite
except ImportError:
    # This works locally on your Windows machine
    from tensorflow import lite as tflite

import numpy as np
import os
import nltk
from nltk.stem import WordNetLemmatizer
import random
import pickle
import json
import numpy as np
import os


lemmatizer = WordNetLemmatizer()

# --- DYNAMIC PATHS ---
# Ensures the app finds files whether running locally or in Docker
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TFLITE_MODEL_PATH = os.path.join(BASE_DIR, "app/ml/chatbot_model.tflite")
WORDS_PATH = os.path.join(BASE_DIR, "app/ml/words.pkl")
CLASSES_PATH = os.path.join(BASE_DIR, "app/ml/classes.pkl")
INTENTS_PATH = os.path.join(BASE_DIR, "app/data/intents.json")

# --- LOAD LITE MODEL ---
# Using the Interpreter instead of load_model() saves ~800MB of RAM
interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- LOAD UTILS ---
words = pickle.load(open(WORDS_PATH, "rb"))
classes = pickle.load(open(CLASSES_PATH, "rb"))
with open(INTENTS_PATH, encoding="utf-8") as f:
    intents = json.load(f)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    # TFLite expects float32 and shape [1, length]
    return np.array([bag], dtype=np.float32)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    
    # Run Inference
    interpreter.set_tensor(input_details[0]['index'], bow)
    interpreter.invoke()
    res = interpreter.get_tensor(output_details[0]['index'])[0]
    
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_bot_response(message: str) -> str:
    intents_list = predict_class(message)
    if not intents_list:
        return "I'm sorry, I don't understand."
    
    tag = intents_list[0]["intent"]
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "I'm not sure how to help with that."