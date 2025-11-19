.PHONY: install train clean format lint

# Variables
PYTHON = python
PIP = pip

install:
	$(PIP) install -r requirements.txt

train:
	$(PYTHON) main.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	rm -rf .pytest_cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info

format:
	black .
	isort .

lint:
	flake8 .
	mypy .
