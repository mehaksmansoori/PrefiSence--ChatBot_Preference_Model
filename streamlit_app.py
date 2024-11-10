import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Trying to unpickle estimator")


import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import joblib
import string 

import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
import pickle  # For loading pre-trained models and tokenizers
import random
import os


# Optionally disable oneDNN custom operations to ensure consistent numerical results
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Resetting the TensorFlow graph if necessary
tf.compat.v1.reset_default_graph()


# Load the trained model
model = joblib.load('trained_model_chatbot.pkl')


# Preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    return ' '.join(tokens)


def preprocess_input(data):
    # Apply preprocessing to inputs
    data_preprocess = preprocess_text(data)
    # Convert texts to sequences
    # Initialize the tokenizer (adjust num_words or other parameters as needed)
    tokenizer_prompt = Tokenizer() 
    data_preprocess_seq = tokenizer_prompt.texts_to_sequences([data_preprocess])

    # Pad sequences
    max_sequence_length = 250
    data_preprocess_seq = pad_sequences(data_preprocess_seq, maxlen=max_sequence_length)

    
    return data_preprocess_seq


# Function to save feedback
#def save_feedback(rating, comments):
 #   feedback_file = 'feedback.csv'
    
    # Check if the file exists
    #if os.path.exists(feedback_file):
     #   df = pd.read_csv(feedback_file)
  #  else:
   #     df = pd.DataFrame(columns=['Rating', 'Comments'])
    
    # Append new feedback
    #new_feedback = pd.DataFrame({'Rating': [rating], 'Comments': [comments]})
    #df = pd.concat([df, new_feedback], ignore_index=True)
    
    # Save to CSV
    #df.to_csv(feedback_file, index=False)


def main():
    st.title("Chatbot Response Alignment Evaluator")

    # User input
    prompt_input = st.text_input("Enter your prompt:")
    
    model_a_input =  st.text_input('Name of first chatbot:', '')
    response_a_input = st.text_area("Enter response from Chatbot A:")

    
    model_b_input = st.text_input('Name of second chatbot:', '')
    response_b_input = st.text_area("Enter response from Chatbot B:")

    # Creating a DataFrame
    df = pd.DataFrame({
        'prompt': [prompt_input],
        'response_a': [response_a_input],
        'response_b': [response_b_input],
        'model_a': [model_a_input],
        'model_b': [model_b_input]
    })

    if st.button("Evaluate"):
        if prompt_input and response_a_input and response_b_input and model_a_input and model_b_input:
            # Preprocess and prepare input
            df['prompt'] = df['prompt'].apply(preprocess_input)
            df['response_a'] = df['response_a'].apply(preprocess_input)
            df['response_b'] = df['response_b'].apply(preprocess_input)
            df['model_a'] = df['model_a'].apply(preprocess_input)
            df['model_b'] = df['model_b'].apply(preprocess_input)
           # Convert features to numpy arrays and ensure proper shape

            X_prompt = np.array(df['prompt'].tolist()).reshape(-1, 1)
            X_response_a = np.array(df['response_a'].tolist()).reshape(-1, 1)
            X_response_b = np.array(df['response_b'].tolist()).reshape(-1, 1)
            X_model_a = np.array(df['model_a'].tolist()).reshape(-1, 1)
            X_model_b = np.array(df['model_b'].tolist()).reshape(-1, 1)

             # Combine all feature arrays horizontally
            X = np.hstack([X_prompt, X_response_a, X_response_b, X_model_a, X_model_b])


            
            # Predict using the loaded model
            prediction = model.predict(X)
            
            
           # Determine the response based on the prediction
            if prediction.all() == 3 :
                st.write(f"Recommended Response: {response_b_input}")
            elif prediction.all() == 2:
                st.write(f"Recommended Response: {response_a_input}")
            else :
                recommended_response = random.choice([response_a_input, response_b_input])
                st.write(f"Recommended Response: {recommended_response}")
          
        else:
            st.write("Please fill in all the fields.")

    # Feedback section
    st.subheader("Feedback")
    rating = st.slider("Rate the model's performance (1-5):", 1, 5)
    comments = st.text_area("Additional comments or suggestions:")
    
    if st.button("Submit Feedback"):
        #save_feedback(rating, comments)
        st.write(f"Rating: {rating}")
        st.write(f"Comments: {comments}")

# Run the main function
if __name__ == "__main__":
    main()







