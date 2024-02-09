VENV := .venv

PROJECT := service
TESTS := tests

IMAGE_NAME := reco_service
CONTAINER_NAME := reco_service

# Prepare

.venv:
	poetry env use python3.9
	poetry install --no-root
	poetry check
	poetry run pip install --upgrade setuptools wheel
	poetry run pip install lightfm==1.17 --no-use-pep517
	poetry run pip install recbole
	poetry run pip install faiss-cpu
	poetry run pip install keras==2.9
	poetry run pip install tensorflow==2.9

setup: .venv


# Clean

clean:
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf $(VENV)


# Format

isort_fix: .venv
	poetry run isort $(PROJECT) $(TESTS)


black_fix:
	poetry run black $(PROJECT) $(TESTS)

format: isort_fix black_fix


# Lint

isort: .venv
	poetry run isort --check $(PROJECT) $(TESTS)

.black:
	poetry run black --check --diff $(PROJECT) $(TESTS)

flake: .venv
	poetry run flake8 $(PROJECT) $(TESTS)

mypy: .venv
	poetry run mypy $(PROJECT) $(TESTS)

pylint: .venv
	poetry run pylint $(PROJECT) $(TESTS)

lint: isort flake mypy pylint


# Test

.pytest:
	poetry run pytest $(TESTS)

test: .venv .pytest


# Docker

build:
	docker build . -t $(IMAGE_NAME)

run: build
	docker run -p 8080:8080 --name $(CONTAINER_NAME) $(IMAGE_NAME)

# All

all: setup format lint test run

.DEFAULT_GOAL = all
