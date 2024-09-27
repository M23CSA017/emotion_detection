import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path: str) -> pd.DataFrame:
    """Load CSV data from the specified file path."""
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Data loaded from {file_path}")
        data.fillna("", inplace=True)
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def load_params(params_path: str) -> int:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        max_features = params["feature_engineering"]["max_features"]
        logging.info(f"Max features loaded: {max_features}")
        return max_features
    except Exception as e:
        logging.error(f"Error loading parameters: {e}")
        raise

def apply_bow(X_train: pd.Series, X_test: pd.Series, max_features: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply Bag of Words transformation."""
    try:
        vectorizer = CountVectorizer(max_features=max_features)
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)
        logging.info("Bag of Words transformation applied.")
        return pd.DataFrame(X_train_bow.toarray()), pd.DataFrame(X_test_bow.toarray())
    except Exception as e:
        logging.error(f"Error applying BoW: {e}")
        raise

def save_features(data_path: str, train_df: pd.DataFrame, test_df: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> None:
    """Save the train and test features along with labels."""
    try:
        os.makedirs(data_path, exist_ok=True)
        train_df["label"] = y_train
        test_df["label"] = y_test
        train_df.to_csv(os.path.join(data_path, "train_bow.csv"), index=False)
        test_df.to_csv(os.path.join(data_path, "test_bow.csv"), index=False)
        logging.info(f"Features saved to {data_path}")
    except Exception as e:
        logging.error(f"Error saving features: {e}")
        raise

def main() -> None:
    try:
        train_data_path = "./data/processed/train_processed.csv"
        test_data_path = "./data/processed/test_processed.csv"
        params_path = "params.yaml"

        train_data = load_data(train_data_path)
        test_data = load_data(test_data_path)

        X_train, y_train = train_data["content"], train_data["sentiment"]
        X_test, y_test = test_data["content"], test_data["sentiment"]

        max_features = load_params(params_path)

        X_train_bow, X_test_bow = apply_bow(X_train, X_test, max_features)

        data_path = os.path.join("data", "features")
        save_features(data_path, X_train_bow, X_test_bow, y_train, y_test)
    except Exception as e:
        logging.error(f"Main execution error: {e}")

if __name__ == "__main__":
    main()
