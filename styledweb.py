import streamlit as st
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title='Sentiment/Emotion Analyzer', page_icon=':smiley:', layout='wide')

st.title('Sentiment/Emotion Analyzer')
st.markdown('---')

# Create two columns for the input and analysis type selection
col1, col2 = st.columns([3, 1])

# Input column
with col1:
    st.write('Enter the text to analyze:')
    user_input = st.text_area('', height=150)
    if not user_input:
        st.warning('Please enter some text.')

# Analysis type column
with col2:
    analysis_type = st.selectbox('Select analysis type', ['Sentiment', 'Emotion'])

# Add some spacing between the input and output sections
st.markdown('---')

# Load the saved vectorizer and model
with open('tfidf_vectorizer2.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

if analysis_type == 'Sentiment':
    with open('modelnaive3.pkl', 'rb') as f:
        model = pickle.load(f)
        img_pos = 'positive.png'
        img_neg = 'negative.png'
        img_neut  = 'neutral.png'
else:
    with open('svm_model2.pkl', 'rb') as f:
        model = pickle.load(f)
        img_sad = 'sad.png'
        img_lov = 'love.png'
        img_hat = 'hate.png'
        img_sup = 'surprise.png'
        img_fer = 'fear.png'
        img_hap = 'happy.png'

# Define preprocessing function
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Add a button to trigger analysis
if st.button('Analyze'):
    if user_input:
        user_input = preprocess_text(user_input)
        if analysis_type == 'Sentiment':
            y_pred = model.predict(np.array([user_input]))[0]
            st.write('The sentiment of the text is:', y_pred)
            if y_pred == 'positive':
                st.image(img_pos, caption='Positive sentiment')
            elif y_pred == 'negative':
                st.image(img_neg, caption='Negative sentiment')
            elif y_pred == 'neutral':
                st.image(img_neut, caption='Neutral sentiment')
        else:
            emotext = vectorizer.transform([user_input])
            y_pred = model.predict(emotext.reshape(1, -1))[0]
            st.write('The emotion of the text is:', y_pred)
            if y_pred == 'happy':
                st.image(img_hap, caption='Happy emotion')
            elif y_pred == 'sadness':
                st.image(img_sad, caption='Sad emotion')
            elif y_pred == 'fear':
                st.image(img_fer, caption='fear emotion')
            elif y_pred == 'love':
                st.image(img_lov, caption='love emotion')
            elif y_pred == 'surprise':
                st.image(img_sup, caption='Surprise emotion')
            elif y_pred == 'anger':
                st.image(img_hat, caption='hate emotion')
                
        # Scroll to the bottom of the page
        scroll_down = """window.scroll(0, document.body.scrollHeight);"""
        st.write("<script>{}</script>".format(scroll_down), unsafe_allow_html=True)