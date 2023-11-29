from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from model import ridge, ridge_input_dict_template, category_columns
import pandas as pd


def get_ridge_input(item_dict: dict) -> pd.DataFrame:
    ridge_input_dict = dict(ridge_input_dict_template)
    ridge_input_dict_keys = list(ridge_input_dict.keys())
    for key, value in item_dict.items():
        if key in category_columns:
            res_key = "_".join((key, str(value)))
            if res_key in ridge_input_dict_keys:
                ridge_input_dict[res_key] = [1]
        else:
            ridge_input_dict[key] = [value]

    return pd.DataFrame(ridge_input_dict)


app = FastAPI()


class Item(BaseModel):
    year: int
    km_driven: int
    mileage: float
    engine: int
    max_power: float
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    seats: int


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    item_dict = item.model_dump()
    ridge_input = get_ridge_input(item_dict)
    prediction = ridge.predict(ridge_input)[0, 0]

    return prediction


@app.post("/predict_items")
def predict_items(items: Items) -> List[float]:
    predictions = []
    for item in items.objects:
        item_dict = item.model_dump()
        ridge_input = get_ridge_input(item_dict)
        prediction = ridge.predict(ridge_input)[0, 0]
        predictions.append(prediction)
    return predictions

