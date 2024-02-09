import pickle
from typing import List

import faiss
import numpy as np


class FaissIndex:
    def __init__(self, user_id_to_uid, iid_to_item_id, users, popular, threshold=0.83, dimension=128, N=10):
        self.dimension = dimension
        self.faiss_index = faiss.IndexFlatL2(self.dimension)
        faiss.omp_set_num_threads(1)
        self.N = N
        self.user_id_to_uid = user_id_to_uid
        self.iid_to_item_id = iid_to_item_id
        self.users = users
        self.popular = popular
        self.threshold = threshold

    def index(self, data):
        assert data.shape[1] == self.dimension, "Размерность векторов должна соответствовать размерности индекса"
        self.faiss_index.add(data)

    def add_popular(self, item_ids: List[int]) -> np.ndarray:
        mask = ~np.isin(self.popular, item_ids)
        filtered_popular = self.popular[mask]
        combined = np.concatenate([item_ids, filtered_popular])
        combined = combined.astype(int)
        result = combined[: self.N]
        return result

    def get_items(self, query_user: int) -> List[int]:
        query_uid = self.user_id_to_uid.get(query_user, None)
        if query_uid is None:
            return self.popular[:10].tolist()
        user_vector = self.users[query_uid]
        user_vector = user_vector.reshape(1, -1)
        assert (
            user_vector.shape[1] == self.dimension
        ), "Размерность векторов запросов должна соответствовать размерности индекса."
        dist, index = self.faiss_index.search(user_vector, self.N)
        filtered_indices = [[iid for iid, dist in zip(index[0], dist[0]) if dist <= self.threshold]]
        item_ids = [self.iid_to_item_id[iid] for iid in filtered_indices]
        item_ids = self.add_popular(item_ids).tolist()
        return item_ids

    def save(self, filepath: str):
        faiss.write_index(self.faiss_index, filepath + "_faiss.index")

        with open(filepath + "_data.pkl", "wb") as f:
            data = {
                "dimension": self.dimension,
                "N": self.N,
                "user_id_to_uid": self.user_id_to_uid,
                "iid_to_item_id": self.iid_to_item_id,
                "users": self.users,
                "popular": self.popular,
                "threshold": self.threshold,
            }
            pickle.dump(data, f)

    @classmethod
    def load(cls, filepath: str) -> "FaissIndex":
        faiss_index = faiss.read_index(filepath + "_faiss.index")

        with open(filepath + "_data.pkl", "rb") as f:
            data = pickle.load(f)

        instance = cls(
            user_id_to_uid=data["user_id_to_uid"],
            iid_to_item_id=data["iid_to_item_id"],
            users=data["users"],
            popular=data["popular"],
            threshold=data["threshold"],
            dimension=data["dimension"],
            N=data["N"],
        )
        instance.faiss_index = faiss_index
        return instance
