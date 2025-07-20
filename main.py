import pandas as pd
from src.preprocess import clean_text
from src.model import train_model
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def main():
    # Load dataset
    df = pd.read_csv('data/spam.csv', encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'message']

    # Clean text
    df['clean_message'] = df['message'].apply(clean_text)

    # Visualize
    sns.countplot(x='label', data=df)
    plt.title('Spam vs Ham Count')
    plt.show()

    # Train model
    clf, vectorizer = train_model(df, text_col='clean_message', label_col='label')

    # ✅ Save model
    joblib.dump(clf, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    print("✅ Model and vectorizer saved.")

if __name__ == "__main__":
    main()
