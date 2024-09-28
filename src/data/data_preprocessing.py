import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from the specified CSV file path."""
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Data loaded from {file_path}")
        data.fillna("", inplace=True)
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def lemmatization(text: str) -> str:
    """Apply lemmatization to the given text."""
    try:
        lemmatizer = WordNetLemmatizer()
        words = text.split()
        lemmatized_text = [lemmatizer.lemmatize(word) for word in words]
        return " ".join(lemmatized_text)
    except Exception as e:
        logging.error(f"Error in lemmatization: {e}")
        raise

def remove_stop_words(text: str) -> str:
    """Remove stop words from the given text."""
    try:
        stop_words = set(stopwords.words("english"))
        filtered_text = [word for word in text.split() if word not in stop_words]
        return " ".join(filtered_text)
    except Exception as e:
        logging.error(f"Error removing stop words: {e}")
        raise

def removing_numbers(text: str) -> str:
    """Remove numbers from the given text."""
    try:
        text_without_numbers = ''.join([char for char in text if not char.isdigit()])
        return text_without_numbers
    except Exception as e:
        logging.error(f"Error removing numbers: {e}")
        raise

def lower_case(text: str) -> str:
    """Convert text to lower case."""
    try:
        return text.lower()
    except Exception as e:
        logging.error(f"Error converting to lower case: {e}")
        raise

def removing_punctuations(text: str) -> str:
    """Remove punctuations from the given text."""
    try:
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        text = re.sub('\s+', ' ', text)  # Remove extra whitespaces
        return text.strip()
    except Exception as e:
        logging.error(f"Error removing punctuations: {e}")
        raise

def removing_urls(text: str) -> str:
    """Remove URLs from the given text."""
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    except Exception as e:
        logging.error(f"Error removing URLs: {e}")
        raise

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the text data in the DataFrame."""
    try:
        df["content"] = df["content"].apply(lower_case)
        df["content"] = df["content"].apply(remove_stop_words)
        df["content"] = df["content"].apply(removing_numbers)
        df["content"] = df["content"].apply(removing_punctuations)
        df["content"] = df["content"].apply(removing_urls)
        df["content"] = df["content"].apply(lemmatization)
        logging.info("Text normalization complete.")
        return df
    except Exception as e:
        logging.error(f"Error normalizing text: {e}")
        raise

def save_processed_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the processed data to a CSV file."""
    try:
        df.to_csv(file_path, index=False)
        logging.info(f"Processed data saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving processed data: {e}")
        raise

def main() -> None:
    try:
        # File paths
        train_data_path = "./data/external/train.csv"
        test_data_path = "./data/external/test.csv"
        interim_data_path = os.path.join("data", "interim")  # Save to data/interim

        # Ensure directories exist
        os.makedirs(interim_data_path, exist_ok=True)

        # Load and process data
        train_data = load_data(train_data_path)
        test_data = load_data(test_data_path)

        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)

        # Save processed data
        save_processed_data(train_processed_data, os.path.join(interim_data_path, "train_processed.csv"))
        save_processed_data(test_processed_data, os.path.join(interim_data_path, "test_processed.csv"))
    except Exception as e:
        logging.error(f"Main execution error: {e}")

if __name__ == "__main__":
    main()
