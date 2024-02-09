import numpy as np
import torch
from recbole.quick_start import load_data_and_model


class MultiVAE:
    def __init__(self, path: str, popular: np.ndarray):
        self.config, self.model, self.dataset, self.train_data, self.valid_data, self.test_data = load_data_and_model(
            model_file=path,
        )
        self.popular = popular

    def add_popular(self, item_ids: np.ndarray, N: int) -> np.ndarray:
        mask = ~np.isin(self.popular, item_ids)
        filtered_popular = self.popular[mask]
        combined = np.concatenate([item_ids, filtered_popular])
        combined = combined.astype(int)
        result = combined[:N]
        return result

    def recommend(self, user_id: int, n: int) -> list:
        try:
            uid_series = self.dataset.token2id(self.dataset.uid_field, [str(user_id)])
        except ValueError:
            recos = self.popular[:n].tolist()
            return recos

        interaction = {self.model.USER_ID: torch.tensor([uid_series])}
        scores = self.model.full_sort_predict(interaction)
        recommended_items = torch.argsort(scores, descending=True)[:n]
        recos = self.dataset.id2token(self.dataset.iid_field, recommended_items)

        if len(recos) < n:
            recos = self.add_popular(recos, n)
        recos = [int(item) for item in recos]
        return recos
