from typing import Any, List

import pandas as pd
from IPython.display import display
from rectools.dataset import Dataset, Interactions


class Visualizer:
    def __init__(self, model: Any, dataset: Dataset, interactions: Interactions, items: pd.DataFrame):
        self.model = model
        self.dataset = dataset
        self.interactions = interactions
        self.items = items

    def visualize(self, users: List, k: int):
        recos = self.model.recommend(users=users, dataset=self.dataset, k=k, filter_viewed=True)
        user_viewed = self.interactions.df[self.interactions.df["user_id"].isin(users)].merge(
            self.items[["title", "genres", "item_id"]], on="item_id", how="left"
        )
        recos = recos.merge(self.items[["title", "genres", "item_id"]], on="item_id")
        return user_viewed, recos

    def visualize_with_color(self, users: List, k: int):
        user_viewed, recos = self.visualize(users, k)
        common_titles = pd.merge(user_viewed[["title"]], recos[["title"]], on="title", how="inner")["title"].unique()

        def highlight_common_movies(row):
            if row["title"] in common_titles:
                return ["background-color: blue"] * len(row)
            return [""] * len(row)

        user_viewed_styled = user_viewed.style.apply(highlight_common_movies, axis=1)
        recos_styled = recos.style.apply(highlight_common_movies, axis=1)

        users_str = ", ".join(map(str, users))

        print("*" * 20 + " " + f"Просмотры пользователей {users_str}" + " " + "*" * 20)
        display(user_viewed_styled)

        print("\n" * 3)

        print("*" * 20 + " " + f"Рекомендации для пользователей {users_str}" + " " + "*" * 20)
        display(recos_styled)

        return user_viewed_styled, recos_styled
