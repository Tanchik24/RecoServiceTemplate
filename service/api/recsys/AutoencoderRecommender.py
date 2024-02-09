import json
from typing import Dict, List, Union

import numpy as np


class AutoencoderRecommender:
    def __init__(
        self,
        X_preds: np.ndarray,
        X_train_and_val: np.ndarray,
        X_test: np.ndarray,
        users_key_dict: Dict[Union[int, str], Union[int, str]],
        items_keys: np.ndarray,
        popular: np.ndarray,
    ) -> None:
        self.X_preds = X_preds
        self.X_train = X_train_and_val
        self.X_test = X_test
        self.X_total = X_train_and_val + X_test
        self.users_key_dict = users_key_dict
        self.items_keys = np.array(items_keys)
        self.popular = popular

    def recommend_items(self, user_id: int, items_to_select_idx: np.ndarray, topn: int = 10) -> np.ndarray:
        user_preds = self.X_preds[user_id][items_to_select_idx]
        items_idx = items_to_select_idx[np.argsort(-user_preds)[:topn]]
        return items_idx

    def evaluate(self) -> Dict[str, float]:
        true_5, true_10 = [], []

        for user_id, _ in enumerate(self.X_test):
            non_zero = np.argwhere(self.X_test[user_id] > 0).ravel()
            all_nonzero = np.argwhere(self.X_total[user_id] > 0).ravel()
            select_from = np.setdiff1d(np.arange(self.X_total.shape[1]), all_nonzero)

            for non_zero_idx in non_zero:
                random_non_interacted_100_items = np.random.choice(select_from, size=100, replace=False)
                preds = self.recommend_items(user_id, np.append(random_non_interacted_100_items, non_zero_idx), topn=10)
                true_5.append(non_zero_idx in preds[:5])
                true_10.append(non_zero_idx in preds)

        return {"recall@5": float(np.mean(true_5)), "recall@10": float(np.mean(true_10))}

    def add_popular(self, item_ids: np.ndarray, N: int) -> np.ndarray:
        mask = ~np.isin(self.popular, item_ids)
        filtered_popular = self.popular[mask]
        combined = np.concatenate([item_ids, filtered_popular])
        combined = combined.astype(int)
        result = combined[:N]
        return result

    def recommend(self, user_id: Union[int, str], topn: int = 10) -> List[int]:
        uid = self.users_key_dict.get(user_id, None)
        if uid is None:
            return self.popular[:topn].tolist()
        all_nonzero = np.argwhere(self.X_total[uid] > 0).ravel()
        select_from = np.setdiff1d(np.arange(self.X_total.shape[1]), all_nonzero)
        preds = self.X_preds[uid][select_from]
        items_idx = select_from[np.argsort(-preds)[:topn]]
        items = [self.items_keys[item_idx] for item_idx in items_idx]
        if len(items) < topn:
            items_array = np.array(items)
            items = self.add_popular(items_array, topn).tolist()
        return items

    def save(self, file_path: str) -> None:
        np.save(file_path + "autoencoder_x_train.npy", self.X_train)
        np.save(file_path + "autoencoder_x_test.npy", self.X_test)
        np.save(file_path + "autoencoder_x_pred.npy", self.X_preds)
        self.users_key_dict = {int(key): item for key, item in self.users_key_dict.items()}
        with open(file_path + "autoencoder_users_key_dict.json", "w", encoding="utf-8") as json_file:
            json.dump(self.users_key_dict, json_file)
        np.save(file_path + "autoencoder_items_keys.npy", self.items_keys)

    @classmethod
    def load(cls, file_path: str) -> "AutoencoderRecommender":
        X_train = np.load(file_path + "autoencoder_x_train.npy")
        X_test = np.load(file_path + "autoencoder_x_test.npy")
        X_preds = np.load(file_path + "autoencoder_x_pred.npy")

        with open(file_path + "autoencoder_users_key_dict.json", "r", encoding="utf-8") as json_file:
            users_key_dict = json.load(json_file)
            users_key_dict = {int(k): int(v) for k, v in users_key_dict.items()}

        items_keys = np.load(file_path + "autoencoder_items_keys.npy")
        popular = np.load(file_path + "popular.npy")

        return AutoencoderRecommender(X_preds, X_train, X_test, users_key_dict, items_keys, popular)
