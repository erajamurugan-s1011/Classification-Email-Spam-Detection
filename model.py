from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def train_model(data, text_col='message', label_col='label'):
    # Convert text to numerical features
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data[text_col])

    # Convert labels to binary: ham=0, spam=1
    y = data[label_col].map({'ham': 0, 'spam': 1})

    # Split dataset (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize and train classifier
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    # Predict on test data
    y_pred = clf.predict(X_test)

    # Print performance metrics
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return clf, vectorizer
