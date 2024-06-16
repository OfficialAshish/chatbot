
from flask import Flask, request, jsonify
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import random, json
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

from  keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

model = tf.keras.models.load_model('utils/mental_health_chatbot.h5')
tokenizer = pickle.load(open('utils/tokenizer.pkl', 'rb'))
lbl_enc = pickle.load(open('utils/label_encoder.pkl', 'rb'))

with open('utils/intents.json', 'r') as f:
    data = json.load(f)

def chat(input_text):
    input_text = input_text.lower()
    input_text = tokenizer.texts_to_sequences([input_text])
    input_text = pad_sequences(input_text, padding='post')
    tag = lbl_enc.inverse_transform([np.argmax(model.predict(input_text))])[0]
    for intent in data['intents']:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            break
    return response

@app.route('/chat', methods=['POST'])
def get_response():
    input_text = request.json['input_text']
    response = chat(input_text)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run()
