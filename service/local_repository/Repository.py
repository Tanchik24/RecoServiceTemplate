import json
import os
import pickle

from dotenv import load_dotenv
from rectools.dataset import Dataset

load_dotenv()


class Repository:
    @staticmethod
    def fetch_popular_model() -> dict:
        file_path = os.getenv("POPULAR")
        root_dir = os.environ.get("ROOT_DIR")
        file_path = os.path.join(root_dir, file_path)
        with open(file_path, "r", encoding="utf-8") as file:
            popular_model = json.load(file)
        return popular_model

    @staticmethod
    def fetch_dataset() -> Dataset:
        file_path = os.getenv("DATASET")
        root_dir = os.environ.get("ROOT_DIR")
        file_path = os.path.join(root_dir, file_path)
        with open(file_path, "rb") as file:
            dataset = pickle.load(file)
        return dataset

    @staticmethod
    def fetch_lightfm_model() -> Dataset:
        file_path = os.getenv("LIGHTFM")
        root_dir = os.environ.get("ROOT_DIR")
        file_path = os.path.join(root_dir, file_path)
        with open(file_path, "rb") as file:
            light_fm = pickle.load(file)
        return light_fm
