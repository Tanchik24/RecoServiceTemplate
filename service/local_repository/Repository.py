import os
import pickle

import numpy as np
from dotenv import load_dotenv

from service.api.recsys.Ranker import Ranker
from service.api.recsys.userknn import UserKnn

load_dotenv()


class Repository:
    def __init__(self) -> None:
        self.root_dir: str = os.path.dirname(os.path.abspath(__file__))
        if self.root_dir is not None:
            parts: list = self.root_dir.split("/")
            index: int = len(parts) - 1 - parts[::-1].index("RecoServiceTemplate")
            self.ROOT_DIR: str = "/".join(parts[: index + 1])
        self.popular: np.ndarray = self.fetch_popular_model()

    def fetch_user_knn_model(self) -> "UserKnn":
        if self.root_dir is None:
            return None
        file_path = os.getenv("KNN")
        file_path = os.path.join(self.ROOT_DIR, file_path)
        with open(file_path, "rb") as file:
            user_knn_model = pickle.load(file)
        return user_knn_model

    def fetch_popular_model(self) -> np.ndarray:
        if self.root_dir is None:
            return None
        file_path = os.getenv("POPULAR")
        file_path = os.path.join(self.ROOT_DIR, file_path)
        popular_model = np.load(file_path)
        return popular_model

    def fetch_ranker_model(self) -> "Ranker":
        if self.root_dir is None:
            return None
        file_path = os.getenv("RANKER")
        file_path = os.path.join(self.ROOT_DIR, file_path)
        ranker_model = Ranker(file_path, self.popular)
        return ranker_model
