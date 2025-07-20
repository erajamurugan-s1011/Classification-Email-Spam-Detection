import streamlit as st
from src.preprocess import clean_text
from src.utils import load_model

# Load the model and vectorizer
model, vectorizer = load_model()

st.title("📩 Email & SMS Spam Detector")
st.write("Enter a message below and find out if it's Spam or Ham.")

# User input
user_input = st.text_area("Message", height=150)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        clean = clean_text(user_input)
        vectorized = vectorizer.transform([clean])
        prediction = model.predict(vectorized)[0]
        result = "📨 Ham (Not Spam)" if prediction == 0 else "🚫 Spam"
        st.success(f"Prediction: {result}")
