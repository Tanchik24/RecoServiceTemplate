import random
from typing import List

from fastapi import APIRouter, FastAPI, Header, Request
from pydantic import BaseModel

from service.api.auth import check_access, check_model_user
from service.local_repository.Repository import Repository
from service.log import app_logger


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


router = APIRouter()
repository = Repository()
ranker_model = repository.fetch_ranker_model()

@router.get(
    path="/health",
    tags=["Health"],
)
async def health() -> str:
    return "I am alive"


@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
)
async def get_reco(model_name: str, user_id: int, request: Request, authorization: str = Header(None)) -> RecoResponse:
    check_access(authorization)
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")

    check_model_user(["dssm_model", "autoencoder_model", "multivae_model", "ranker_model"], model_name, user_id)

    k_recs = request.app.state.k_recs

    if ranker_model is None:
        recos = [random.randint(0, 100) for _ in range(k_recs)]
    elif model_name == 'ranker_model':
        recos = ranker_model.recommend(user_id)

    return RecoResponse(user_id=user_id, items=recos)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
