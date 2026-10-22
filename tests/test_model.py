
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "price_model.pkl"
)

FEATURE_NAMES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


def load_model():
    return joblib.load(MODEL_PATH)


def test_model_loads():
    model = load_model()
    assert model is not None


def test_model_prediction_is_valid():
    model = load_model()

    sample = pd.DataFrame(
        [[
            8.3252,
            41.0,
            6.984127,
            1.02381,
            322.0,
            2.555556,
            37.88,
            -122.23,
        ]],
        columns=FEATURE_NAMES,
    )

    prediction = model.predict(sample)

    assert len(prediction) == 1
    assert np.isfinite(prediction[0])
    assert prediction[0] > 0
