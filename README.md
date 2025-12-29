# 📩 Spam Message Detection

This project is a **Machine Learning and NLP-based web application** that classifies SMS or text messages as **Spam** or **Not Spam (Ham)**.  
The application is built using **Streamlit**, allowing users to instantly check whether a message is safe or suspicious.

---

## 📌 Features
- Text preprocessing (lowercasing, cleaning, stopword removal)
- TF-IDF vectorization for feature extraction
- Machine Learning-based spam classification
- Interactive and user-friendly **Streamlit UI**
- Displays prediction confidence score
- Deployment-safe NLP pipeline (no tokenizer dependency issues)

---

## 🛠️ Tech Stack
- **Python**
- **NLTK** (Text preprocessing)
- **scikit-learn** (Machine Learning)
- **TF-IDF Vectorizer**
- **Streamlit** (Web Application)
- **Joblib** (Model persistence)

---

## 📂 Project Structure

-spam-message-detection/
-│
-├── app.py
-├── Model.pkl
-├── Vectorizer.pkl
-├── requirements.txt
-├── README.md


---

## 🌐 Live Demo
🔗 Click here to try the app  
👉 *https://spam-message-detection-qy6ye6hb8ghptq7cuhteuw.streamlit.app/*

---

## ⚙️ How It Works
1. User enters an SMS or text message
2. Text is cleaned and preprocessed
3. TF-IDF converts text into numerical features
4. ML model predicts spam or not spam
5. Result and confidence score are displayed

---

## 📊 Results & Performance
- Achieved strong accuracy using **TF-IDF + Naive Bayes**
- Effectively detects promotional, phishing, and scam messages
- Performs well on short and real-world SMS text
- Successfully deployed on **Streamlit Cloud**

---

## 🚀 Future Enhancements
- BERT-based spam classification
- URL and phishing link detection
- Multi-language spam detection
- Detailed evaluation metrics (Precision, Recall, F1-Score)

---

## ✨ Author
👤 **Siddiqui Usman Ahmed Siraj Ahmed**

📧 siddiquiusman915256@gmail.com  

🔗 [LinkedIn Profile](https://www.linkedin.com/in/usman-siddiqui-948006347)

---

⭐ If you find this project useful, please consider giving it a star!
