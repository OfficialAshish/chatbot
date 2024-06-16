from flask_cors import CORS 
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pickle, json, random
from  keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.preprocessing.sequence import pad_sequences

model = tf.keras.models.load_model('utils/mental_health_chatbot.h5')
tokenizer = pickle.load(open('utils/tokenizer.pkl', 'rb'))
lbl_enc = pickle.load(open('utils/label_encoder.pkl', 'rb'))
with open('utils/intents.json', 'r') as f:
    data = json.load(f)
    
    
# Create a Flask app
app = Flask(__name__)
CORS(app)

@app.route('/chat', methods=['POST'])
def chatbot():
  # Get the user input from the request
  input_text = request.json['input_text']
  # Preprocess the input text
  input_text = input_text.lower()
  input_text = tokenizer.texts_to_sequences([input_text])
  input_text = pad_sequences(input_text, padding='post', maxlen=18)
  
  # print(input_text)
  # print("Shape after tokenization:", input_text.shape)
  # print("Shape after padding:", input_text.shape) 
  
    
  # Predict the tag
  tag = lbl_enc.inverse_transform([np.argmax(model.predict(input_text))])[0]

  # Get the response based on the tag
  for intent in data['intents']:
    if intent['tag'] == tag:
      response = random.choice(intent['responses'])
      break

  # Return the response
  return jsonify({'response': response})

if __name__ == '__main__':
  app.run()
