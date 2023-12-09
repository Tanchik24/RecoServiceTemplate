import json
import os
import pickle

from dotenv import load_dotenv

from service.api.recsys.userknn import UserKnn

load_dotenv()


class Repository:
    @staticmethod
    def fetch_user_knn_model() -> "UserKnn":
        file_path = os.getenv("KNN")
        root_dir = os.environ.get("ROOT_DIR")
        file_path = os.path.join(root_dir, file_path)
        with open(file_path, "rb") as file:
            user_knn_model = pickle.load(file)
        return user_knn_model

    @staticmethod
    def fetch_popular_model() -> dict:
        file_path = os.getenv("POPULAR")
        root_dir = os.environ.get("ROOT_DIR")
        file_path = os.path.join(root_dir, file_path)
        with open(file_path, "r", encoding="utf-8") as file:
            popular_model = json.load(file)
        return popular_model
