import time
from pprint import pprint
from typing import Dict, List

import pandas as pd
from rectools import Columns
from rectools.dataset import Interactions
from rectools.metrics import calc_metrics
from rectools.model_selection import TimeRangeSplitter
from tqdm.auto import tqdm


class CrossValScore:
    def __init__(self, models: Dict, metrics: Dict, splitter: TimeRangeSplitter, interactions: Interactions):
        self.fold_iterator = None
        self.models = models
        self.metrics = metrics
        self.splitter = splitter
        self.interactions = interactions

    def init(self, train_ids: List, test_ids: List):
        df_train = self.interactions.df.iloc[train_ids].copy()
        df_test = self.interactions.df.iloc[test_ids][Columns.UserItem].copy()
        catalog = df_train[Columns.Item].unique()
        return df_train, df_test, catalog

    def evaluate(self, n_splits: int = 3):
        self.fold_iterator = self.splitter.split(self.interactions, collect_fold_stats=True)
        results = []

        for train_ids, test_ids, fold_info in tqdm(self.fold_iterator, total=n_splits):
            print(f"\n==================== Fold {fold_info['i_split']}")
            pprint(fold_info)

            df_train, df_test, catalog = self.init(train_ids, test_ids)

            for model_name, model in self.models.items():
                start_time = time.time()
                model.fit(df_train)
                fit_time = time.time() - start_time

                recos = model.predict(df_test)

                metric_values = calc_metrics(
                    self.metrics, reco=recos, interactions=df_test, prev_interactions=df_train, catalog=catalog
                )

                res = {"fold": fold_info["i_split"], "model": model_name, "training_time": fit_time}
                res.update(metric_values)
                results.append(res)

        df = pd.DataFrame(results).groupby(["model"]).mean().reset_index()
        df = df.drop(columns="fold")
        return df
