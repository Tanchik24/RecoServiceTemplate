import pickle
from typing import List

import numpy as np


class Ranker:
    def __init__(self, path: str, popular: np.ndarray, n=10):
        with open(path, "rb") as file:
            self.ranker = pickle.load(file)
        self.popolar = 0
        self.n = n
        self.popular = popular

    def add_popular(self, item_ids: List, N: int) -> np.ndarray:
        mask = ~np.isin(self.popular, item_ids)
        filtered_popular = self.popular[mask]
        combined = np.concatenate([item_ids, filtered_popular])
        combined = combined.astype(int)
        result = combined[:N]
        return result

    def recommend(self, user_id: int) -> List[int]:
        try:
            recos = self.ranker[self.ranker.user_id == user_id].item_id.tolist()[0]
        except IndexError:
            recos = self.add_popular([], self.n)

        if len(recos) < self.n:
            recos = self.add_popular(recos, self.n)
        return recos
