.PHONY: train run demo test docker-up

train:
	python3 -m backend.app.cli train

demo:
	python3 -m backend.app.cli demo

run:
	uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

test:
	python3 -m unittest discover -s tests -v

docker-up:
	docker compose up --build
