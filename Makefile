PYTHON ?= python
AAAI_KIT ?= ../AAAI_AuthorKit27

.PHONY: all submission experiments real-data figures test paper supplement checklist audit clean

all: experiments test paper

submission: test experiments real-data paper checklist audit

experiments:
	PYTHONPATH=experiments $(PYTHON) experiments/exact_risk.py --output-dir results/exact_pilot
	PYTHONPATH=experiments $(PYTHON) experiments/monte_carlo.py --output-dir results/monte_carlo_pilot
	PYTHONPATH=experiments $(PYTHON) experiments/raw_horizon.py --output-dir results/raw_horizon_pilot
	PYTHONPATH=experiments $(PYTHON) experiments/noise_regimes.py --output-dir results/noise_regimes
	PYTHONPATH=experiments $(PYTHON) experiments/heavy_tail_horizon.py --a 2 --b 5.5 --r 3 --model-size 65536 --horizon-powers 5 7 9 11 13 --trials 3000 --fit-points 3 --output-dir results/heavy_tail_horizon
	PYTHONPATH=experiments $(PYTHON) experiments/dense_features.py --model-size 512 --block-powers 6 8 10 12 14 --trials 120 --fit-points 3 --output-dir results/dense_features
	PYTHONPATH=experiments $(PYTHON) experiments/dense_ar_control.py --dimension 256 --n-values 16 24 36 54 80 120 176 --trials 40 --fit-points 4 --output-dir results/dense_ar_control
	PYTHONPATH=experiments $(PYTHON) experiments/matched_ess_compute.py --output-dir results/matched_ess_compute
	$(MAKE) figures

real-data:
	PYTHONPATH=experiments $(PYTHON) experiments/real_sequential.py --output-dir results/real_sequential
	$(MAKE) figures

figures:
	PYTHONPATH=experiments $(PYTHON) experiments/paper_figures.py

test:
	PYTHONPATH=experiments $(PYTHON) -m unittest discover -s experiments -p 'test_*.py' -v

paper: figures
	cd paper && TEXINPUTS=".:$(AAAI_KIT):" BSTINPUTS=".:$(AAAI_KIT):" latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
	$(MAKE) supplement

supplement:
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex

checklist:
	$(PYTHON) tools/build_checklist.py --template AAAI_AuthorKit27/ReproducibilityChecklist.tex --output paper/reproducibility_checklist.tex
	cd paper && TEXINPUTS=".:$(AAAI_KIT):" latexmk -pdf -interaction=nonstopmode -halt-on-error reproducibility_checklist.tex

audit:
	$(PYTHON) tools/audit_submission.py --paper-dir paper --output results/final_audit/build_report.json

clean:
	cd paper && latexmk -C main.tex && latexmk -C supplement.tex && latexmk -C reproducibility_checklist.tex
	rm -rf experiments/__pycache__ tools/__pycache__ results/final_audit
	rm -f paper/aaai2027.sty paper/aaai2027.bst
