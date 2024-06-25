import string
import unicodedata
from pyvi import ViTokenizer
import numpy as np
import os

import pandas as pd
from collections import Counter

class Data_Processor():
    def __init__(
        self,
        sentiment_path: str,
        sent_path: str,
        stopwords_path: str = 'vietnamese-stopwords.txt'
    ):
        self.sentiment_path = sentiment_path
        self.sent_path = sent_path
        self.stopwords_path = stopwords_path
        self.stopwords = self.load_stopwords()

    def load_stopwords(self):
        try:
            with open(self.stopwords_path, 'r', encoding='utf-8') as sf:
                stopwords = [word.strip() for word in sf.readlines()]
            return stopwords
        except FileNotFoundError:
            print(f"Stopwords file '{self.stopwords_path}' not found.")
            return []

    def preprocess_text(self, text):
        # Lowercase the text
        text = text.lower()

        # Remove diacritics (accents)
        text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('utf-8')

        # Remove punctuation
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)

        # Tokenize using Vietnamese tokenizer (pyvi)
        text = ViTokenizer.tokenize(text)

        # Remove stopwords
        text = ' '.join([word for word in text.split() if word not in self.stopwords])

        return text

    def read_txt_files(self):
        try:
            with open(self.sentiment_path, 'r', encoding='utf-8') as sentiment_file:
                sentiment_data = [int(line.strip()) for line in sentiment_file.readlines()]
            
            with open(self.sent_path, 'r', encoding='utf-8') as sent_file:
                sent_data = [self.preprocess_text(line.strip()) for line in sent_file.readlines()]

            return sentiment_data, sent_data

        except FileNotFoundError:
            print("File not found. Please check the file paths.")
            return None, None

    def create_dataframe(self):
        sentiment_data, sent_data = self.read_txt_files()
        
        if sentiment_data is None or sent_data is None:
            return None
        
        # Count occurrences of each sentiment label
        sentiment_counts = Counter(sentiment_data)

        # Initialize lists for dataframe creation
        text_column = sent_data
        negative_column = [1 if label == 0 else 0 for label in sentiment_data]
        neutral_column = [1 if label == 1 else 0 for label in sentiment_data]
        positive_column = [1 if label == 2 else 0 for label in sentiment_data]

        # Create DataFrame
        df = pd.DataFrame({
            'Text': text_column,
            'Negative': negative_column,
            'Neutral': neutral_column,
            'Positive': positive_column
        })

        return df


