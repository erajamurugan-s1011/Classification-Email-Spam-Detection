import joblib

def save_model(model, vectorizer, model_path='model.pkl', vectorizer_path='vectorizer.pkl'):
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

def load_model(model_path='model.pkl', vectorizer_path='vectorizer.pkl'):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer
