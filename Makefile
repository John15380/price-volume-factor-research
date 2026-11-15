PYTHON ?= python

.PHONY: all experiment test report
all: experiment report

experiment:
	$(PYTHON) code2_new.py --with-xgboost

test:
	$(PYTHON) -B code2_new.py --self-test

report:
	$(PYTHON) -B code2_new.py --report
