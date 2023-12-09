from typing import List

from fastapi import APIRouter, FastAPI, Header, Request
from pydantic import BaseModel

from service.api.auth import check_access
from service.api.exceptions import ModelNotFoundError, UserNotFoundError
from service.local_repository.Repository import Repository
from service.log import app_logger


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


user_knn_model = Repository.fetch_user_knn_model()
popular_model = Repository.fetch_popular_model()

router = APIRouter()


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

    if model_name != "knn_model":
        raise ModelNotFoundError(error_message=f"Model {model_name} not found")
    if user_id > 10**9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    k_recs = request.app.state.k_recs
    knn_rec = user_knn_model.recommend(user_id)[: int(k_recs * 0.5)]
    pop_rec = popular_model.get(str(user_id), popular_model["all"])[: int(k_recs - len(knn_rec))]
    recos = knn_rec + pop_rec
    return RecoResponse(user_id=user_id, items=recos)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
