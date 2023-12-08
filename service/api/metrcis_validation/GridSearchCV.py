from implicit.nearest_neighbours import BM25Recommender, CosineRecommender, TFIDFRecommender
from sklearn.model_selection import ParameterGrid

from service.api.metrcis_validation.metrics import CrossValScore
from service.api.recsys.userknn import UserKnn


class GridSearchCV:
    def __init__(self, cross_val_score: CrossValScore, param_grid: dict):
        self.cross_val_score = cross_val_score
        self.param_grid = param_grid
        self.best_params_ = None
        self.best_score_ = 0

    def search(self, n_splits: int = 3):
        grid = ParameterGrid(self.param_grid)

        for params in grid:
            print(f"Testing parameters: {params}")
            recommender_type = params["model"]
            K = params["K"]

            if recommender_type == CosineRecommender:
                model = UserKnn(CosineRecommender(K=K))
            elif recommender_type == TFIDFRecommender:
                model = UserKnn(TFIDFRecommender(K=K))
            elif recommender_type == BM25Recommender:
                model = UserKnn(BM25Recommender(K=K))
            else:
                raise ValueError("Invalid recommender type")

            self.cross_val_score.models = {"custom_model": model}

            df_result = self.cross_val_score.evaluate(n_splits=n_splits)
            score = df_result["MAP@10"][0]
            if score > self.best_score_:
                self.best_score_ = score
                self.best_params_ = params
