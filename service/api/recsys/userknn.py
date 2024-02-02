from collections import Counter
from typing import Any, List

import numpy as np
import pandas as pd
import scipy as sp
from implicit.nearest_neighbours import ItemItemRecommender


class UserKnn:
    """Class for fit-perdict UserKNN model
    based on ItemKNN model from implicit.nearest_neighbours
    """

    def __init__(self, model: ItemItemRecommender):
        self.item_idf = pd.DataFrame()
        self.users: Any = None
        self.weights_matrix: Any = None
        self.user_knn: Any = None
        self.watched: Any = None
        self.interaction_matrix: Any = None
        self.items_mapping: Any = None
        self.items_inv_mapping: Any = None
        self.users_mapping: Any = None
        self.users_inv_mapping: Any = None
        self.model = model
        self.is_fitted = False

    def get_mappings(self, train: pd.DataFrame):
        self.users_inv_mapping = dict(enumerate(train["user_id"].unique()))
        self.users_mapping = {v: k for k, v in self.users_inv_mapping.items()}

        self.items_inv_mapping = dict(enumerate(train["item_id"].unique()))
        self.items_mapping = {v: k for k, v in self.items_inv_mapping.items()}

    def get_matrix(
        self, df: pd.DataFrame, user_col: str = "user_id",
        item_col: str = "item_id", weight_col: str = None
    ):
        if weight_col:
            weights = df[weight_col].astype(np.float32)
        else:
            weights = np.ones(len(df), dtype=np.float32)

        self.interaction_matrix = sp.sparse.coo_matrix(
            (weights, (df[item_col].map(self.items_mapping.get),
                       df[user_col].map(self.users_mapping.get)))
        )

        self.watched = (
            df.groupby(user_col, as_index=False).agg({item_col: list}).rename(
                columns={user_col: "sim_user_id"})
        )

        return self.interaction_matrix

    def idf(self, n: int, x: float) -> float:
        return np.log((1 + n) / (1 + x) + 1)

    def _get_users(self, train: pd.DataFrame):
        self.users = set(train["user_id"])

    def _count_item_idf(self, df: pd.DataFrame, n: int):
        item_cnt = Counter(df["item_id"].values)
        temp_item_idf = pd.DataFrame.from_dict(item_cnt, orient="index", columns=["doc_freq"]).reset_index()
        temp_item_idf["idf"] = temp_item_idf["doc_freq"].apply(lambda x: self.idf(n, x))  # pylint: disable=E1136,E1137
        self.item_idf = temp_item_idf

    def fit(self, train: pd.DataFrame):
        self._get_users(train)
        self.user_knn = self.model
        self.get_mappings(train)
        self.weights_matrix = self.get_matrix(train)

        n = train.shape[0]
        self._count_item_idf(train, n)

        self.user_knn.fit(self.weights_matrix)
        self.is_fitted = True

    def _generate_recs_mapper(self, model: ItemItemRecommender, N: int = 50):
        def _recs_mapper(user):
            user_id = self.users_mapping[user]
            users, sim = model.similar_items(user_id, N=N)
            return [self.users_inv_mapping[user] for user in users], sim

        return _recs_mapper

    def predict(self, test: pd.DataFrame, N_recs: int = 10) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Please call fit before predict")

        mapper = self._generate_recs_mapper(model=self.user_knn)

        recs = pd.DataFrame({"user_id": test["user_id"].unique()})
        recs["sim_user_id"], recs["sim"] = zip(*recs["user_id"].map(mapper))
        recs = recs.set_index("user_id").apply(pd.Series.explode).reset_index()

        recs = (
            recs[~(recs["user_id"] == recs["sim_user_id"])]
            .merge(self.watched, on=["sim_user_id"], how="left")
            .explode("item_id")
            .sort_values(["user_id", "sim"], ascending=False)
            .drop_duplicates(["user_id", "item_id"], keep="first")
            .merge(self.item_idf, left_on="item_id", right_on="index",
                   how="left")
        )
        recs["score"] = recs["sim"] * recs["idf"]
        recs = recs.sort_values(["user_id", "score"], ascending=False)
        recs["rank"] = recs.groupby("user_id").cumcount() + 1
        return recs[recs["rank"] <= N_recs][
            ["user_id", "item_id", "score", "rank"]]

    def recommend(self, user: int, N_recs: int = 10) -> List:
        if user not in self.users:
            return []

        if not self.is_fitted:
            raise ValueError("Please call fit before predict")

        sim_users, sims = self._generate_recs_mapper(model=self.user_knn)(user)

        sim_users_np = np.array(sim_users)
        sims_np = np.array(sims)

        mask = sim_users_np != user
        sim_users_np = sim_users_np[mask]
        sims_np = sims_np[mask]

        watched_items = self.watched.set_index("sim_user_id").loc[sim_users_np]["item_id"]
        watched_items_np = np.concatenate(watched_items.values)

        item_idf = self.item_idf.set_index("index").loc[watched_items_np][
            "idf"].values

        scores = np.repeat(sims_np, [len(items) for items in
                                     watched_items.values]) * item_idf

        sorted_indices = np.argsort(-scores)
        sorted_items = watched_items_np[sorted_indices]

        _, unique_indices = np.unique(sorted_items, return_index=True)
        unique_top_items = sorted_items[np.sort(unique_indices)]

        top_items = unique_top_items[:N_recs]

        return top_items
