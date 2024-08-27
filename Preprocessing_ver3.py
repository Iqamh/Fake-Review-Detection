import pandas as pd
import nltk
import re
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')


# Load data
data = pd.read_csv(
    'C:\\Users\\iqbal\\Downloads\\Penting\\TA\\New folder\\MachineLearning\\FinalLabel_Balance.csv', delimiter=';')


# Function to translate text to Indonesian, ignoring English words
def translate_to_indonesian(text):
    try:
        if detect(text) != 'id':
            translator = GoogleTranslator(source='auto', target='id')
            translated_text = translator.translate(text)
            return translated_text
    except LangDetectException:
        pass  # If language detection fails, return the original text
    return text


# Translate 'Review' column to Indonesian
data['Translated_Review'] = data['Review'].apply(translate_to_indonesian)


# Read abbreviation mappings from a text file
def load_abbreviation_mappings(filepath):
    with open(filepath, 'r') as file:
        abbreviation_mapping = json.load(file)
    return abbreviation_mapping


# Example abbreviation file path
abbreviation_mapping = load_abbreviation_mappings(
    'C:\\Users\\iqbal\\Downloads\\Penting\\TA\\New folder\\MachineLearning\\combined_slang_words.txt')


# Function to replace abbreviations with their full form
def replace_abbreviations(text, abbreviation_mapping):
    tokens = word_tokenize(text)

    new_tokens = []
    for token in tokens:
        split_tokens = re.split(r'[^\w\s]', token)
        cleaned_tokens = [re.sub(r'\d+', '', word) for word in split_tokens]
        new_tokens.extend(cleaned_tokens)
    tokens = new_tokens

    replaced_tokens = [abbreviation_mapping.get(word, word) for word in tokens]
    return ' '.join(replaced_tokens)


# Function to handle repeated characters and map to root form
def handle_repeated_characters(word):
    word = re.sub(r'(.)\1+$', r'\1', word)
    return word


# Custom stopwords list excluding negations
def custom_stopwords():
    sastrawi_stopwords_factory = StopWordRemoverFactory()
    sastrawi_stopwords = set(sastrawi_stopwords_factory.get_stop_words())
    nltk_stopwords = set(stopwords.words('indonesian'))
    important_tokens = {"tidak", "bukan", "belum", "jangan", "baik", "lama", "biasa", "kurang", "cukup", "benar", "tinggi", "baru", "tanpa"}
    return (sastrawi_stopwords | nltk_stopwords) - important_tokens

custom_stop_words = custom_stopwords()


# Define preprocessing function
def preprocess_text(text, abbreviation_mapping):
    # Tokenization
    tokens = word_tokenize(text)

    # Handle punctuation-separated words
    new_tokens = []
    for token in tokens:
        split_tokens = re.split(r'[^\w\s]', token)
        cleaned_tokens = [re.sub(r'\d+', '', word) for word in split_tokens]
        new_tokens.extend(cleaned_tokens)
    tokens = new_tokens

    # Lowercasing
    tokens = [word.lower() for word in tokens]

    # Remove numbers if necessary
    tokens = [re.sub(r'\d+', '', word) for word in tokens]

    # Replace abbreviations
    tokens = [abbreviation_mapping.get(word, word) for word in tokens]

    # Remove punctuation and whitespace
    tokens = [re.sub(r'[^\w\s]', '', word) for word in tokens]
    tokens = [word.strip() for word in tokens if word.strip()]

    # Handle repeated characters
    tokens = [handle_repeated_characters(word) for word in tokens]

    # Stemming - Sastrawi
    stemmer = StemmerFactory().create_stemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    # Remove custom stopwords
    tokens = [word for word in tokens if word not in custom_stop_words]

    return tokens


# Preprocess the translated 'Review' column
data['Tokenized_Review'] = data['Translated_Review'].apply(
    lambda x: preprocess_text(x, abbreviation_mapping))

# Save the preprocessed data
data.to_csv('C:\\Users\\iqbal\\Downloads\\Penting\\TA\\New folder\\MachineLearning\\Final_Label_Preprocessed_Balance_Final-Plis.csv', index=False, sep=';')

print("Preprocessing complete and data saved.")
