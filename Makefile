.PHONY: setup data train evaluate serve test monitor retrain

setup:
	python -m pip install -e ".[dev]"

data:
	python -m dvc repro validate_raw

train:
	python -m dvc repro train

evaluate:
	python -m src.evaluation.final_evaluation

serve:
	python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000

test:
	python -m pytest -q

monitor:
	@echo "Monitoring is planned for a future module."

retrain:
	@echo "Controlled retraining is planned for a future module."
