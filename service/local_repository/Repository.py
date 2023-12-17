import json
import os
import pickle

from dotenv import load_dotenv
from lightfm import LightFM
from rectools.dataset import Dataset

from service.api.recsys.userknn import UserKnn

load_dotenv()


class Repository:
    def __init__(self) -> None:
        root_dir: str = os.path.dirname(os.path.abspath(__file__))
        parts: list = root_dir.split("/")
        index: int = parts.index("RecoServiceTemplate") + 1
        self.ROOT_DIR: str = "/".join(parts[:index])

    def fetch_user_knn_model(self) -> "UserKnn":
        file_path = os.getenv("KNN")
        file_path = os.path.join(self.ROOT_DIR, file_path)
        with open(file_path, "rb") as file:
            user_knn_model = pickle.load(file)
        return user_knn_model

    def fetch_popular_model(self) -> dict:
        file_path = os.getenv("POPULAR")
        file_path = os.path.join(self.ROOT_DIR, file_path)
        with open(file_path, "r", encoding="utf-8") as file:
            popular_model = json.load(file)
        return popular_model

    def fetch_dataset(self) -> "LightFM":
        file_path = os.getenv("DATASET")
        file_path = os.path.join(self.ROOT_DIR, file_path)
        with open(file_path, "rb") as file:
            dataset = pickle.load(file)
        return dataset

    def fetch_lightfm_model(self) -> "Dataset":
        file_path = os.getenv("LIGHTFM")
        file_path = os.path.join(self.ROOT_DIR, file_path)
        with open(file_path, "rb") as file:
            light_fm = pickle.load(file)
        return light_fm
