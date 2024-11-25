import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Function for text preprocessing
stem = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Cache the dataset loading function
@st.cache_data
def load_data():
    news_dataset = pd.read_csv('C:/Users/HP/Desktop/Fake News Detection/train.csv')
    news_dataset = news_dataset.fillna('')
    news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']
    news_dataset['content'] = news_dataset['content'].apply(stemming)
    return news_dataset

# Cache the model training function
@st.cache_resource
def train_model(X_train, Y_train):
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    return model

# Load and preprocess the dataset
data = load_data()

# Prepare data for training
X = data['content'].values
Y = data['label'].values

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X_tfidf, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the model
logreg_model = train_model(X_train, Y_train)

# Streamlit App Interface
st.title("Fake News Detector")

st.subheader("Select an Index from the Dataset")
index = st.number_input("Enter an index number:", min_value=0, max_value=len(data) - 1, step=1)

if st.button("Classify"):
    # Fetch and preprocess the content for the selected index
    content = data.iloc[index]['content']
    content_vectorized = tfidf_vectorizer.transform([content])

    # Predict
    prediction = logreg_model.predict(content_vectorized)
    result = "Real" if prediction[0] == 0 else "Fake"

    # Display Result
    st.write(f"News Content: **{data.iloc[index]['title']}**")
    st.write(f"Prediction: **{result}**")

# Accuracy Display
st.write("\n---")
st.write("Model Evaluation:")
accuracy = accuracy_score(Y_test, logreg_model.predict(X_test))
st.write(f"Accuracy Score: **{accuracy:.2f}**")
