from flask import Flask, request, jsonify
from flask_cors import CORS 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import pickle
import requests
import json

app = Flask(__name__)
CORS(app)

@app.route('/find_similarity', methods=['POST'])
def find_similarity():
    user_data = request.json['user_data']

    df = pickle.load(open('df_combined.pkl', 'rb'))
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['tokens'])
    # user_str = " ".join(user_data)
    user_str = ""
    for i in user_data:
        user_str += i['text']
            
    user_vector = tfidf_vectorizer.transform([user_str])
    similarities = cosine_similarity(user_vector, tfidf_matrix)
    most_similar_index = similarities.argmax()

    return jsonify({'id': df['_id'][most_similar_index]})

@app.route('/filter', methods=['POST'])
def searchExpert():
    user_data = request.json['data']
    df = pickle.load(open('df_expertise.pkl', 'rb'))
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['tokens'])
    
    user_vector = tfidf_vectorizer.transform([user_data])
    similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()
    
    # Set a similarity threshold 
    threshold = 0.2
    similar_indices = [i for i, similarity in enumerate(similarities) if similarity >= threshold]
    
    similar_ids = df['_id'].iloc[similar_indices].tolist()
    
    return jsonify({'ids': similar_ids})


@app.route('/summarize_chat', methods=['POST'])
def summarize_chat():
    chat_json = request.json['chat_json']
    
    print(chat_json['messages'][0])

    chat_text = "\n".join([f"{msg['sender']}: {msg['message']}" for msg in chat_json['messages'][0]])
    prompt = f"Summarize the important points from the following conversation:\n\n{chat_text}"
    
    headers = {"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiMWM4MjRhYWMtNTU1MC00ODZlLThkOWItODY3MGFjZTY3ZTQ1IiwidHlwZSI6ImFwaV90b2tlbiJ9.BDVBInIe67BdIa82HRCTm18ri6tWYbfvWAQ0Pe2GvM4"}
    url = "https://api.edenai.run/v2/text/summarize"
    payload = {
        "providers": "openai",
        "language": "en",
        "text": prompt,
        "output_sentences": 6,
    }
    
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        summary = result['openai']['result']
        return jsonify({'summary': summary}), 200
    else:
        return jsonify({'error': 'Failed to summarize chat'}), response.status_code
 
  
import tensorflow as tf
import numpy as np
import random
from  tensorflow.keras.preprocessing.sequence import pad_sequences

model = tf.keras.models.load_model('model/utils/mental_health_chatbot.h5')
tokenizer = pickle.load(open('model/utils/tokenizer.pkl', 'rb'))
lbl_enc = pickle.load(open('model/utils/label_encoder.pkl', 'rb'))
with open('model/utils/intents.json', 'r') as f:
    data = json.load(f)
     
     
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
    app.run(debug=True, port=8888)
