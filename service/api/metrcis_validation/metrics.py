import time
from pprint import pprint
from typing import Dict, List

import numpy as np
import pandas as pd
from rectools import Columns
from rectools.dataset import Dataset, Interactions
from rectools.metrics import calc_metrics
from rectools.model_selection import TimeRangeSplitter
from tqdm.auto import tqdm


class CrossValScore:
    def __init__(
        self,
        models: Dict,
        metrics: Dict,
        splitter: TimeRangeSplitter,
        interactions: Interactions,
        users_features: pd.DataFrame,
        items_features: pd.DataFrame,
    ):
        self.fold_iterator = None
        self.models = models
        self.metrics = metrics
        self.splitter = splitter
        self.interactions = interactions
        self.users_features = users_features
        self.items_features = items_features

    def init(self, train_ids: List, test_ids: List):
        df_train = self.interactions.df.iloc[train_ids]
        items_features = self.items_features.loc[self.items_features["id"].isin(df_train[Columns.Item])].copy()
        users_features = self.users_features.loc[self.users_features["id"].isin(df_train[Columns.User])].copy()
        dataset = Dataset.construct(
            df_train,
            user_features_df=users_features,
            cat_user_features=list(users_features["feature"].unique()),
            item_features_df=items_features,
            cat_item_features=list(items_features["feature"].unique()),
        )
        df_test = self.interactions.df.iloc[test_ids][Columns.UserItem]
        cold_users = set(df_test[Columns.User]) - set(df_train[Columns.User])
        df_test.drop(df_test[df_test[Columns.User].isin(cold_users)].index, inplace=True)
        test_users = np.unique(df_test[Columns.User])
        catalog = df_train[Columns.Item].unique()
        return dataset, df_train, df_test, test_users, catalog

    def evaluate(self, k: int = 10, n_splits: int = 3):
        self.fold_iterator = self.splitter.split(self.interactions, collect_fold_stats=True)
        results = []

        for train_ids, test_ids, fold_info in tqdm(self.fold_iterator, total=n_splits):
            print(f"\n==================== Fold {fold_info['i_split']}")
            pprint(fold_info)

            dataset, df_train, df_test, test_users, catalog = self.init(train_ids, test_ids)

            for model_name, model in self.models.items():
                start_time = time.time()
                model.fit(dataset)
                fit_time = time.time() - start_time

                recos = model.recommend(users=test_users, dataset=dataset, k=k, filter_viewed=True)

                metric_values = calc_metrics(
                    self.metrics, reco=recos, interactions=df_test, prev_interactions=df_train, catalog=catalog
                )

                res = {"fold": fold_info["i_split"], "model": model_name, "training_time": fit_time}
                res.update(metric_values)
                results.append(res)

        df = pd.DataFrame(results).groupby(["model"]).mean().reset_index()
        df = df.drop(columns="fold")
        return df
