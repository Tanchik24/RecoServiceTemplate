import os
from typing import List

from dotenv import load_dotenv

from service.api.exceptions import ModelNotFoundError, UserNotFoundError

from .exceptions import InvalidAuthorization

load_dotenv()


def check_access(authorization: str):
    if authorization is None:
        raise InvalidAuthorization(error_message="No token")
    token = authorization.split(" ")[-1]
    secret_key = os.getenv("SECRET_KEY")
    print(secret_key)
    if token != secret_key:
        raise InvalidAuthorization(error_message="Invalid token")


def check_model_user(true_model_names: List, model: str, user_id: int) -> None:
    if model not in true_model_names:
        raise ModelNotFoundError(error_message=f"Model {model} not found")
    if user_id > 10**9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")
