import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# load trained model
bayes = joblib.load('naive_bayes_meessages (1).pkl')


# load tfidf vectorizer
tfidf_vec = joblib.load('tfidf_vectorizer.pkl')

# data transform
def preprocess_text(text):
    return tfidf_vec.transform([text])

# classify messages
def classify_message(model, text):
    test_input = preprocess_text(text)
    result = model.predict(test_input)[0]
    if result == 1:
        return "Normal Message"
    elif result == 0:
        return "SPAM"

# Title and description
st.title("Message Classifier")
st.write("This app classifies messages as either SPAM or Normal.")

# Input text box
text_input = st.text_input("Enter a message:")

# Classify button
# streamlit run --server.address 10.1.44.156 streamlit.py
if st.button("Classify"):
    if text_input:
        result = classify_message(bayes, text_input)
        if result == "Normal Message":
            st.success("Result: Normal Message")
        elif result == "SPAM":
            st.error("Result: SPAM")
    else:
        st.warning("Please enter a message.")