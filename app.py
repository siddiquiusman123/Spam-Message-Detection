import streamlit as st
import pandas as pd
import joblib
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def download_nltk():
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)

download_nltk()


st.title("📩 Spam Message Detection App")
st.write("Check whether an SMS or message is **Spam** or **Not Spam**.")

st.markdown("**✉️ Enter your message here:**")

user_input = st.text_area(
    "Message Input",
    placeholder="Type your message here...",
    height=120,
    label_visibility="collapsed"
)



stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words("english"))

def preprocess(text):

    text = text.lower() 
    words = word_tokenize(text)

    preprocessed_text = [

        stemmer.stem(word)
        for word in words
        if word.isalpha() and word not in stop_words
    ]

    return " ".join(preprocessed_text)

model = joblib.load("Model.pkl")
vectorizer = joblib.load("Vectorizer.pkl")

# clean_text = preprocess(user_input)
# vec = vectorizer.transform([clean_text])
# prediction = model.predict(vec)
# print("Spam 🚫" if prediction[0] == 1 else "Not Spam ✅")

if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message")
    else:
        clean_text = preprocess(user_input)
        vector = vectorizer.transform([clean_text])
        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector).max()

        if prediction == 1:
            st.error(f"🚫 Spam Message\n\nConfidence : {probability*100:.2f}%")
        else:
            st.success(f"✅ Not Spam Message\n\nConfidence : {probability*100:.2f}%")