import pickle
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random # Import the random module 

# Load the trained model and tokenizer
model = load_model('utils/mental_health_chatbot.h5')
tokenizer = pickle.load(open('utils/tokenizer.pkl', 'rb'))
lbl_enc = pickle.load(open('utils/label_encoder.pkl', 'rb'))
with open('utils/intents.json', 'r') as f:
    data = json.load(f)
    
def chat():
    while True:
        # Get user input
        input_text = input("You: ")

        # Check if the user wants to exit BEFORE preprocessing 
        if input_text.lower() == 'quit': 
            break

        # Preprocess the input text
        input_text = input_text.lower()
        input_text = tokenizer.texts_to_sequences([input_text])
        input_text = pad_sequences(input_text, padding='post')

        # Predict the tag
        tag = lbl_enc.inverse_transform([np.argmax(model.predict(input_text))])[0]

        # Get the response based on the tag
        for intent in data['intents']:
            if intent['tag'] == tag:
                response = random.choice(intent['responses'])
                break

        # Print the response
        print("Bot:", response)

# Start the chat
chat()