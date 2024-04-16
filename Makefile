# Simple Makefile to automate some procedures

all: build

build:
	@python3 setup.py sdist bdist_wheel

test:
	@python3 -m unittest discover -v -s subspyces/tests -p '*_test.py'

clean:
	@rm -rf build dist *.egg-info __pycache__