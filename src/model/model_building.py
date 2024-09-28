import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path: str) -> pd.DataFrame:
    """Load CSV data from the specified file path."""
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Data loaded from {file_path}")
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)["model_building"]
        logging.info(f"Model parameters loaded: {params}")
        return params
    except Exception as e:
        logging.error(f"Error loading parameters: {e}")
        raise

def train_model(X_train: pd.DataFrame, y_train: pd.Series, params: dict) -> GradientBoostingClassifier:
    """Train a Gradient Boosting Classifier model."""
    try:
        clf = GradientBoostingClassifier(n_estimators=params["n_estimators"], learning_rate=params["learning_rate"])
        clf.fit(X_train, y_train)
        logging.info("Model training complete.")
        return clf
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise

def save_model(model: GradientBoostingClassifier, file_path: str) -> None:
    """Save the trained model as a pickle file."""
    try:
        with open(file_path, "wb") as file:
            pickle.dump(model, file)
        logging.info(f"Model saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise

def main() -> None:
    try:
        data_path = "./data/features/train_bow.csv"
        params_path = "params.yaml"
        model_save_path = "models/model.pkl"

        train_data = load_data(data_path)
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        params = load_params(params_path)
        clf = train_model(X_train, y_train, params)
        save_model(clf, model_save_path)
    except Exception as e:
        logging.error(f"Main execution error: {e}")

if __name__ == "__main__":
    main()
