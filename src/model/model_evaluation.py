import pandas as pd
import pickle
import json
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
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

def load_model(model_path: str) -> GradientBoostingClassifier:
    """Load the trained model from a pickle file."""
    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        logging.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def evaluate_model(model: GradientBoostingClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate the model and calculate metrics."""
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }
        logging.info(f"Model evaluation metrics: {metrics}")
        return metrics
    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        with open(file_path, "w") as file:
            json.dump(metrics, file, indent=4)
        logging.info(f"Metrics saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving metrics: {e}")
        raise

def main() -> None:
    try:
        test_data_path = './data/features/test_bow.csv'
        model_path = './models/model.pkl'
        metrics_save_path = 'metrics.json'

        test_data = load_data(test_data_path)
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        model = load_model(model_path)
        metrics = evaluate_model(model, X_test, y_test)
        save_metrics(metrics, metrics_save_path)
    except Exception as e:
        logging.error(f"Main execution error: {e}")

if __name__ == "__main__":
    main()
    