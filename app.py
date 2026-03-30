import streamlit as st
import joblib
import nltk
import re

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet

# --------------------------------------------------
# NLTK DOWNLOAD (CACHED)
# --------------------------------------------------
@st.cache_resource
def download_nltk():
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("omw-1.4")  # for lemmatizer

download_nltk()

# --------------------------------------------------
# LOAD MODEL & VECTORIZER (CACHED)
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("Model.pkl")
    vectorizer = joblib.load("Vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_artifacts()

# --------------------------------------------------
# NLP TOOLS
# --------------------------------------------------
lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words("english"))
stop_words = stop_words - {"not", "no", "nor", "never"}

# --------------------------------------------------
# TEXT PREPROCESSING (DEPLOYMENT SAFE)
# --------------------------------------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-z\s]", "", text)        # remove non-letters

    tokens = text.split()   # simple tokenization

    processed_words = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(processed_words)

# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------
st.set_page_config(
    page_title="Spam Message Detection",
    page_icon="📩",
    layout="centered"
)

st.title("📩 Spam Message Detection App")
st.write("Check whether an SMS or message is **Spam** or **Not Spam**.")

user_input = st.text_area(
    "✉️ Enter your message",
    placeholder="Type your message here...",
    height=130
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message.")
    else:
        clean_text = preprocess(user_input)

        if clean_text.strip() == "":
            st.warning("⚠️ Message contains only stopwords or invalid text.")
        else:
            with st.spinner("Analyzing message..."):
                vector = vectorizer.transform([clean_text])
                prediction = model.predict(vector)[0]

            if prediction == 1:
                st.error("🚫 **Spam Message**")
            else:
                st.success("✅ **Not Spam Message**")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption("NLP | TF-IDF | Machine Learning | Streamlit")