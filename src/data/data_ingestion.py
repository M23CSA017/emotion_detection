import os
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_params(params_path: str) -> float:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        test_size = params["data_ingestion"]["test_size"]
        logging.info(f"Test size loaded: {test_size}")
        return test_size
    except Exception as e:
        logging.error(f"Error loading parameters: {e}")
        raise

def read_data(url: str) -> pd.DataFrame:
    """Read data from a URL."""
    try:
        df = pd.read_csv(url)
        logging.info(f"Data read successfully from {url}")
        return df
    except Exception as e:
        logging.error(f"Error reading data: {e}")
        raise

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process the data by removing unnecessary columns and filtering rows."""
    try:
        df.drop(columns=["tweet_id"], inplace=True)
        final_df = df[df["sentiment"].isin(["happiness", "sadness"])]
        final_df["sentiment"].replace({"happiness": 1, "sadness": 0}, inplace=True)
        logging.info("Data processing complete.")
        return final_df
    except Exception as e:
        logging.error(f"Error processing data: {e}")
        raise

def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    """Save train and test data to the specified path."""
    try:
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
        logging.info(f"Data saved to {data_path}")
    except Exception as e:
        logging.error(f"Error saving data: {e}")
        raise

def main() -> None:
    try:
        params_path = "params.yaml"
        test_size = load_params(params_path)
        url = "https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv"
        df = read_data(url)
        final_df = process_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        data_path = os.path.join("data", "raw")
        save_data(data_path, train_data, test_data)
    except Exception as e:
        logging.error(f"Main execution error: {e}")

if __name__ == "__main__":
    main()
