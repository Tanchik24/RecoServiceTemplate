import time
from typing import Any

import nmslib
import numpy as np


class ANNRecommendation:
    def __init__(self, model, dataset, M=48, efc=100, num_threads=6, K=10, space_name="negdotprod") -> None:
        self.index_time_params = {"M": M, "indexThreadQty": num_threads, "efConstruction": efc, "post": 0}
        self.query_time_params = {"efSearch": efc}
        self.model = model
        self.dataset = dataset
        self.num_threads = num_threads
        self.K = K
        self.space_name = space_name
        self.augmented_item_embeddings: Any = None
        self.augmented_user_embeddings: Any = None
        self.index: Any = None

    def get_vectors(self) -> tuple:
        user_embeddings, item_embeddings = self.model.get_vectors(self.dataset)
        user_shape, item_shape = user_embeddings.shape, item_embeddings.shape
        print(f"Размер эмбединга для юзеров: {user_shape} \n Размер эмбединга для айтемо: {item_shape}")
        return user_embeddings, item_embeddings

    def augment_inner_product(self, user_embeddings, item_embeddings) -> None:
        normed_factors = np.linalg.norm(item_embeddings, axis=1)
        max_norm = normed_factors.max()

        extra_dim = np.sqrt(max_norm**2 - normed_factors**2).reshape(-1, 1)
        self.augmented_item_embeddings = np.append(item_embeddings, extra_dim, axis=1)

        extra_zero = np.zeros((user_embeddings.shape[0], 1))
        self.augmented_user_embeddings = np.append(user_embeddings, extra_zero, axis=1)

    def create_index(self) -> None:
        # pylint: disable=c-extension-no-member
        self.index = nmslib.init(method="hnsw", space=self.space_name, data_type=nmslib.DataType.DENSE_VECTOR)
        self.index.addDataPointBatch(self.augmented_item_embeddings)
        self.index.createIndex(self.index_time_params)

    def create_query_params(self) -> None:
        self.index.setQueryTimeParams(self.query_time_params)

    def fit(self) -> None:
        user_embeddings, item_embeddings = self.get_vectors()
        self.augment_inner_product(user_embeddings, item_embeddings)
        self.create_index()
        self.create_query_params()

    def get_recommendation(self, users) -> list:
        users_intermal_ids = self.dataset.user_id_map.convert_to_internal(users)
        query_matrix = self.augmented_user_embeddings[users_intermal_ids, :]
        query_qty = query_matrix.shape[0]
        start = time.time()
        nbrs = self.index.knnQueryBatch(query_matrix, k=self.K, num_threads=self.num_threads)
        end = time.time()
        print(
            f"""kNN time total=%f {end - start}, per
            query=%f {float(end - start) / query_qty, self.num_threads * float(end - start) },
             per query adjusted for thread number=%f {query_qty}"""
        )
        results = nbrs[0][0]
        items = self.dataset.item_id_map.convert_to_external(results)
        return items
