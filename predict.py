import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.preprocess import clean_text
from src.utils import load_model

def predict_spam(text):
    model, vectorizer = load_model()
    clean = clean_text(text)
    vectorized = vectorizer.transform([clean])
    prediction = model.predict(vectorized)[0]
    return "Spam" if prediction == 1 else "Ham"

if __name__ == "__main__":
    user_input = input("Enter an email or SMS message:\n>> ")
    result = predict_spam(user_input)
    print(f"\nPrediction: {result}")
