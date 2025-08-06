import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords once (will skip if already downloaded)
nltk.download('stopwords')

# Load English stopwords set and initialize stemmer
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def clean_text(text):
    """
    Clean input text by:
    - converting to lowercase
    - removing non-alphabetical characters
    - removing stopwords
    - applying stemming
    
    Args:
        text (str): Raw input text string
    
    Returns:
        str: Cleaned and preprocessed text string
    """
    text = text.lower()
    text = re.sub(r'[^a-z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [ps.stem(w) for w in words]
    return ' '.join(words)
