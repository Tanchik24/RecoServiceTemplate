from collections import Counter

import numpy as np
import pandas as pd
import scipy as sp
from implicit.nearest_neighbours import ItemItemRecommender


class UserKnn:
    """Class for fit-perdict UserKNN model
    based on ItemKNN model from implicit.nearest_neighbours
    """

    def __init__(self, model: ItemItemRecommender, users, N_users: int = 50):
        self.N_users = N_users
        self.model = model
        self.users = users

    def get_mappings(self, train: pd.DataFrame):
        self.users_inv_mapping = dict(enumerate(train["user_id"].unique()))
        self.users_mapping = {v: k for k, v in self.users_inv_mapping.items()}

        self.items_inv_mapping = dict(enumerate(train["item_id"].unique()))
        self.items_mapping = {v: k for k, v in self.items_inv_mapping.items()}

    def _get_users_dict(self, train: pd.DataFrame):
        user_items = train.groupby("user_id")["item_id"].agg(list)

        self.users_dict = {
            user.user_id: [(user.age, user.sex), user_items.loc[user.user_id]]
            for user in self.users.itertuples()
            if user.user_id in user_items.index
        }

    def get_matrix(
        self, df: pd.DataFrame, user_col: str = "user_id", item_col: str = "item_id", weight_col: str = None
    ):
        if weight_col:
            weights = df[weight_col].astype(np.float32)
        else:
            weights = np.ones(len(df), dtype=np.float32)

        self.interaction_matrix = sp.sparse.coo_matrix(
            (weights, (df[item_col].map(self.items_mapping.get), df[user_col].map(self.users_mapping.get)))
        )

        self.watched = (
            df.groupby(user_col, as_index=False).agg({item_col: list}).rename(columns={user_col: "sim_user_id"})
        )
        self.watched = {user.sim_user_id: set(user.item_id) for user in self.watched.itertuples()}

    def idf(self, n: int, x: float):
        return np.log((1 + n) / (1 + x) + 1)

    def _count_item_idf(self, df: pd.DataFrame):
        item_cnt = Counter(df["item_id"].values)
        item_idf = pd.DataFrame.from_dict(item_cnt, orient="index", columns=["doc_freq"]).reset_index()
        item_idf["idf"] = item_idf["doc_freq"].apply(lambda x: self.idf(self.n, x))
        self.item_idf = item_idf

    def _get_train_dfs(self, train: pd.DataFrame):
        dfs = []
        train = train.merge(self.users, on="user_id", how="left")
        train = train.dropna(axis=0, how="any")
        for sex in list(self.users["sex"].unique()):
            for age in list(self.users["age"].unique()):
                df = train[(train["sex"] == sex) & (train["age"] == age)]
                dfs.append(((age, sex), df))
        return dfs

    def _generate_recs_mapper(self, model: ItemItemRecommender, user_mapping, user_inv_mapping, N: int):
        def _recs_mapper(user):
            user_id = user_mapping[user]
            users, sim = model.similar_items(user_id, N=N)
            return [user_inv_mapping[user] for user in users], sim

        return _recs_mapper

    def fit(self, train: pd.DataFrame):
        self._get_users_dict(train)
        self.n = train.shape[0]
        dfs = self._get_train_dfs(train)
        self._count_item_idf(train)
        self.rec_dict = {}

        for i, (info, df) in enumerate(dfs):
            if df.empty:
                continue
            print(f"{i} обучение для {info}")
            self.get_mappings(df)
            self.get_matrix(df)
            self.model.fit(self.interaction_matrix)

            mapper = self._generate_recs_mapper(
                model=self.model,
                user_mapping=self.users_mapping,
                user_inv_mapping=self.users_inv_mapping,
                N=self.N_users,
            )

            recs = pd.DataFrame({"user_id": df["user_id"].unique()})
            recs["sim_user_id"], recs["sim"] = zip(*recs["user_id"].map(mapper))

            self.rec_dict[info] = {user.user_id: [user.sim_user_id, user.sim] for user in recs.itertuples()}

    def predict(self, user, N_recs: int = 10):
        user_info, user_items = self.users_dict.get(user, (None, list()))
        user_items = set(user_items)

        if user_info is None:
            return []

        sim_user_list = self.rec_dict[user_info][user]
        df = pd.DataFrame({"sim_users": sim_user_list[0], "sim": sim_user_list[1]})
        print(user_items)
        df["item_id"] = df["sim_users"].apply(lambda x: set(self.users_dict.get(x, (None, list()))[1]))
        df = df.explode("item_id")
        df = df.sort_values("sim", ascending=False).merge(
            self.item_idf, left_on="item_id", right_on="index", how="left"
        )
        df["score"] = df["sim"] * df["idf"]
        df = df.sort_values("score", ascending=False)

        df["rank"] = [i for i in range(1, df.shape[0] + 1)]
        return list(df[df["rank"] <= N_recs]["item_id"])
