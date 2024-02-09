import os
import pickle

import numpy as np
from dotenv import load_dotenv

from service.api.recsys.AutoencoderRecommender import AutoencoderRecommender
from service.api.recsys.FaissIndex import FaissIndex
from service.api.recsys.MultiVAE import MultiVAE
from service.api.recsys.userknn import UserKnn

load_dotenv()


class Repository:
    def __init__(self) -> None:
        self.root_dir: str = os.path.dirname(os.path.abspath(__file__))
        if self.root_dir is not None:
            parts: list = self.root_dir.split("/")
            index: int = len(parts) - 1 - parts[::-1].index("RecoServiceTemplate")
            self.ROOT_DIR: str = "/".join(parts[: index + 1])

    def fetch_user_knn_model(self) -> "UserKnn":
        if self.root_dir is None:
            return None
        file_path = os.getenv("KNN")
        if file_path is None:
            return None
        file_path = os.path.join(self.ROOT_DIR, file_path)
        with open(file_path, "rb") as file:
            user_knn_model = pickle.load(file)
        return user_knn_model

    def fetch_popular_model(self) -> dict:
        if self.root_dir is None:
            return None
        file_path = os.getenv("POPULAR")
        if file_path is None:
            return None
        file_path = os.path.join(self.ROOT_DIR, file_path)
        popular_model = np.load(file_path, allow_pickle=True)
        return popular_model

    def fetch_dssm_model(self) -> "FaissIndex":
        if self.root_dir is None:
            return None
        dssm_path = os.getenv("DSSM")
        if dssm_path is None:
            return None
        dssm_path = os.path.join(self.ROOT_DIR, dssm_path)
        dssm = FaissIndex.load(dssm_path)
        return dssm

    def fetch_autoencoder_model(self) -> "AutoencoderRecommender":
        if self.root_dir is None:
            return None
        au_path = os.getenv("AUTOENCODER")
        if au_path is None:
            return None
        au_path = os.path.join(self.ROOT_DIR, au_path)
        autoencoder = AutoencoderRecommender.load(au_path)
        return autoencoder

    def fetch_multivae_model(self) -> "MultiVAE":
        if self.root_dir is None:
            return None
        popular = np.array(self.fetch_popular_model())
        mul_path = os.getenv("MULTIVAE")
        if mul_path is None:
            return None
        mul_path = os.path.join(self.ROOT_DIR, mul_path)
        model = MultiVAE(mul_path, popular)
        return model
