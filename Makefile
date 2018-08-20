SHELL = bash

init:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest glyph --doctest-modules
	pytest tests --cov=glyph --cov-config setup.cfg

integration:
	pytest tests/integration_test --runslow -n8 --cov=glyph --cov-config setup.cfg

dev:
	pip install -r requirements-dev.txt

doc:
	make -C doc clean
	make -C doc html

environment:
	python doc/make_envyml.py
