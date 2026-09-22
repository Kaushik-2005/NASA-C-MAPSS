.PHONY: setup data train evaluate serve test monitor retrain

setup:
	python -m pip install -e ".[dev]"

data:
	@echo "Module 3 will validate FD001 data."

train:
	python -m src.training.train

evaluate:
	@echo "Evaluation pipeline is introduced in Module 7."

serve:
	@echo "API service is introduced in Module 11."

test:
	python -m pytest -q

monitor:
	@echo "Monitoring pipeline is introduced in Module 14."

retrain:
	@echo "Retraining pipeline is introduced in Module 15."
