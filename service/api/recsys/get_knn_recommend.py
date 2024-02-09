from typing import List, Optional

import numpy as np

from service.local_repository.Repository import Repository

repository = Repository()
knn = repository.fetch_user_knn_model()
pop = np.array(repository.fetch_popular_model())


def get_knn_rocommend(user: int, N_recs: int = 10) -> Optional[List[int]]:
    if knn is None:
        return None
    knn_rec = np.array(knn.recommend(user)[: int(N_recs * 0.5)])
    mask = ~np.isin(pop, knn_rec)
    filtered_popular = pop[mask]
    combined = np.concatenate([knn_rec, filtered_popular])
    recos = combined[:N_recs]
    return recos.tolist()
