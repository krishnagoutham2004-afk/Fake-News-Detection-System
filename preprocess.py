import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords
nltk.download('stopwords')

# Initialize stemmer
stemmer = PorterStemmer()

# Load stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Split text into words
    words = text.split()

    # Remove stopwords and apply stemming
    filtered_words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(filtered_words)