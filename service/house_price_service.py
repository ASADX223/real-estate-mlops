
from pathlib import Path

import bentoml
import pandas as pd
from pydantic import BaseModel, Field


PROJECT_DIR = Path(__file__).resolve().parent.parent

# Load the latest registered model
model_tag = bentoml.sklearn.get(
    "real_estate_price_model:latest"
)
model = model_tag.load_model()


class HouseFeatures(BaseModel):
    MedInc: float = Field(gt=0)
    HouseAge: float = Field(ge=0)
    AveRooms: float = Field(gt=0)
    AveBedrms: float = Field(gt=0)
    Population: float = Field(ge=0)
    AveOccup: float = Field(gt=0)
    Latitude: float = Field(ge=-90, le=90)
    Longitude: float = Field(ge=-180, le=180)


@bentoml.service(name="real_estate_price_service")
class HousePriceService:

    @bentoml.api
    def predict(self, data: HouseFeatures) -> dict:
        features = pd.DataFrame(
            [data.model_dump()]
        )

        prediction = model.predict(features)[0]

        return {
            "predicted_median_house_value_usd": round(
                float(prediction) * 100000, 2
            )
        }
