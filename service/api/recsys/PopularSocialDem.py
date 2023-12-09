from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd
from rectools.dataset import Interactions
from scipy.sparse import csr_matrix
from scipy.stats import mode


class PopularSocialDem:
    users_dict: dict
    item_set_covered: List[Any]
    sparse_matrix: csr_matrix
    inv_item_mapping: dict
    inv_user_mapping: dict
    item_id_mapping: Dict[Any, Any]
    user_id_mapping: Dict[Any, Any]
    top_dict_soc_dem: dict
    interactions: pd.DataFrame

    def __init__(self, users):
        self.interactions_df = None
        self.users = users

    def _get_topk_dict_soc_dem(self) -> None:
        interactions = self.interactions_df.merge(self.users, on="user_id", how="left")
        self.interactions = interactions.dropna(axis=0, how="any")

        counts = (
            self.interactions.groupby(["age", "sex", "item_id"])
            .size()
            .reset_index(name="count")
            .sort_values(["age", "sex", "count"], ascending=False)
        )
        self.top_dict_soc_dem = counts.groupby(["age", "sex"])["item_id"].apply(list).to_dict()

    def _get_spars_matrix(self) -> None:
        user_id_codes = self.interactions_df["user_id"].astype("category").cat.codes
        item_id_codes = self.interactions_df["item_id"].astype("category").cat.codes

        self.user_id_mapping = dict(zip(self.interactions_df["user_id"], user_id_codes))
        self.item_id_mapping = dict(zip(self.interactions_df["item_id"], item_id_codes))

        self.inv_user_mapping = {v: k for k, v in self.user_id_mapping.items()}
        self.inv_item_mapping = {v: k for k, v in self.item_id_mapping.items()}

        self.sparse_matrix = csr_matrix(
            (self.interactions_df["weight"], (user_id_codes, item_id_codes)), dtype=np.float32
        )

    def _get_top_items_covered_users(self) -> None:
        assert self.sparse_matrix.format == "csr"
        self.item_set_covered = []
        covered_users: np.ndarray = np.zeros(self.sparse_matrix.shape[0], dtype=bool)
        while covered_users.sum() != len(covered_users):
            top_item = mode(self.sparse_matrix[~covered_users].indices)[0]
            self.item_set_covered.append(top_item)
            covered_users += np.maximum.reduceat(
                self.sparse_matrix.indices == top_item, self.sparse_matrix.indptr[:-1], dtype=bool
            )
        self.item_set_covered = [self.inv_item_mapping[item] for item in self.item_set_covered]

    def _get_users_dict(self) -> None:
        user_items = self.interactions_df.groupby("user_id")["item_id"].agg(set)
        self.users_dict = {
            user.user_id: [(user.age, user.sex), user_items.get(user.user_id, set())]
            for user in self.users.itertuples()
        }

    def fit(self, interactions: Interactions):
        self.interactions_df = interactions
        self._get_users_dict()

        self._get_topk_dict_soc_dem()
        self._get_spars_matrix()
        self._get_top_items_covered_users()

    def pred_for_one_user(self, user: int, n_rec: int) -> Set:
        user_info, user_items = self.users_dict.get(user, (None, []))
        results: Set[int] = set()

        potential_recommendations = self.top_dict_soc_dem.get(user_info, self.item_set_covered)

        for rec in potential_recommendations:
            if rec not in user_items and len(results) < n_rec:
                results.add(rec)

        def can_add_rec(rec: int, user_items: Any, results: set, n_rec: int) -> bool:
            return rec not in user_items and rec not in results and len(results) < n_rec

        if len(results) < n_rec:
            for rec in self.item_set_covered:
                if can_add_rec(rec, user_items, results, n_rec):
                    results.add(rec)
        return results

    def predict(self, user: Any, n_rec: int = 20, df: bool = True) -> Any:
        if df is False:
            results = self.pred_for_one_user(user, n_rec)
            return results

        recs = pd.DataFrame({"user_id": user["user_id"].unique()})
        recs["item_id"] = recs["user_id"].apply(lambda x: self.pred_for_one_user(x, n_rec))
        recs = recs.explode("item_id")
        recs["rank"] = recs.groupby("user_id").cumcount() + 1
        return recs
