init:
	pip install -r requirements.txt
	pip install -e .

test: dev
	py.test tests/unittest --cov=glyph

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
	make -C doc clean
	sphinx-apidoc -T -d 1 -o doc/source/api glyph
	make -C doc html

pypi: dev
	rm -rf dist/*
	python setup.py sdist bdist_wheel
	twine upload dist/*
