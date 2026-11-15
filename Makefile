PYTHON ?= python

.PHONY: all experiment test report package
all: experiment report package

experiment:
	$(PYTHON) code2_new.py --with-xgboost

test:
	mkdir -p results/logs
	$(PYTHON) -m unittest discover -s tests -p 'test_code2*.py' -v > results/logs/tests.log 2>&1

report: test
	$(PYTHON) report/build_report.py
	mkdir -p tmp/pdfs/current/build output/pdf
	cd report && xelatex -interaction=nonstopmode -halt-on-error -jobname=report -output-directory=../tmp/pdfs/current/build report_ready.tex > ../results/logs/xelatex_pass1.log 2>&1
	cd report && xelatex -interaction=nonstopmode -halt-on-error -jobname=report -output-directory=../tmp/pdfs/current/build report_ready.tex > ../results/logs/xelatex_pass2.log 2>&1
	cp tmp/pdfs/current/build/report.pdf output/pdf/report.pdf

package:
	$(PYTHON) tools/package_submission.py
