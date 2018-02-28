SHELL = bash

init:
	pip install -r requirements.txt
	pip install -e .

test:
	py.test tests --cov=glyph --cov-config setup.cfg

integration:
	py.test tests --runslow -n8 --cov=glyph --cov-config setup.cfg

dev:
	pip install -r requirements-dev.txt

doc:
	make -C doc clean
	make -C doc html

environment:
	python doc/make_envyml.py
