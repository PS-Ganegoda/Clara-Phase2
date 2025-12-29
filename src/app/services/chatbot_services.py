import os
import json
import pickle
import random
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# -----------------------------
# Environment fixes
# -----------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF logs

# -----------------------------
# NLTK data
# -----------------------------
lemmatizer = WordNetLemmatizer()
nltk.download("punkt")
nltk.download("wordnet")

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "ml/chatbot_model.h5")
WORDS_PATH = os.path.join(BASE_DIR, "ml/words.pkl")
CLASSES_PATH = os.path.join(BASE_DIR, "ml/classes.pkl")
INTENTS_PATH = os.path.join(BASE_DIR, "data/intents.json")

# -----------------------------
# Load static data
# -----------------------------
words = pickle.load(open(WORDS_PATH, "rb"))
classes = pickle.load(open(CLASSES_PATH, "rb"))
intents = json.load(open(INTENTS_PATH))

# -----------------------------
# Lazy load model
# -----------------------------
model = None

def get_model():
    global model
    if model is None:
        model = load_model(MODEL_PATH)
    return model

# -----------------------------
# Chatbot functions
# -----------------------------
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(w.lower()) for w in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = get_model().predict(np.array([bow]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [{"intent": classes[i], "probability": str(r)} 
               for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x["probability"], reverse=True)
    return results

def get_bot_response(message: str) -> str:
    intents_list = predict_class(message)
    if not intents_list:
        return "Sorry, I didn’t understand that."
    tag = intents_list[0]["intent"]
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Sorry, something went wrong."
