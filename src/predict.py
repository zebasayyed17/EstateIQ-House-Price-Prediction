
"""Load the trained house price model and generate predictions."""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "data" / "models" / "random_forest.pkl"
PREPROCESSOR_PATH = PROJECT_ROOT / "data" / "models" / "preprocessor.pkl"


def predict(input_data: dict) -> float:
    """Predict property listing price in Indian rupees."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Random Forest model not found. Run the model training script first."
        )

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            "Preprocessor not found. Run the feature preparation script first."
        )

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    frame = pd.DataFrame([input_data])

    transformed_data = preprocessor.transform(frame)

    if hasattr(transformed_data, "toarray"):
        transformed_data = transformed_data.toarray()

    prediction_log = model.predict(transformed_data)[0]

    predicted_price = np.expm1(prediction_log)

    return float(max(0, predicted_price))
