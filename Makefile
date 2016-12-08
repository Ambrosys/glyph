init:
	pip install -r requirements.txt
	pip install -e .

test: dev
	py.test tests/unittest

integration: dev
	py.test tests --runslow -n8

dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -e .

freeze:
	pip install pip-tools
	pip-compile --output-file requirements.txt requirements-to-freeze.txt

doc: dev
	rm -rf doc/build
	sphinx-apidoc -o doc/source/api glyph
	make -C doc html
