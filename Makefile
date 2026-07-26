PYTHON ?= python
AAAI_KIT ?= ../AAAI_AuthorKit27

.PHONY: all experiments test paper supplement clean

all: experiments test paper

experiments:
	PYTHONPATH=experiments $(PYTHON) experiments/exact_risk.py --output-dir results/exact_pilot
	PYTHONPATH=experiments $(PYTHON) experiments/monte_carlo.py --output-dir results/monte_carlo_pilot
	PYTHONPATH=experiments $(PYTHON) experiments/raw_horizon.py --output-dir results/raw_horizon_pilot
	mkdir -p paper/figures
	cp results/exact_pilot/learning_curves.pdf paper/figures/learning_curves.pdf
	cp results/exact_pilot/phase_collapse.pdf paper/figures/phase_collapse.pdf
	cp results/monte_carlo_pilot/monte_carlo_validation.pdf paper/figures/monte_carlo_validation.pdf
	cp results/raw_horizon_pilot/raw_horizon_validation.pdf paper/figures/raw_horizon_validation.pdf

test:
	PYTHONPATH=experiments $(PYTHON) -m unittest discover -s experiments -p 'test_*.py' -v

paper:
	cd paper && TEXINPUTS=".:$(AAAI_KIT):" BSTINPUTS=".:$(AAAI_KIT):" latexmk -e '$$bibtex = q/bibtex.original %O %B/' -pdf -interaction=nonstopmode -halt-on-error main.tex
	$(MAKE) supplement

supplement:
	cd paper && TEXINPUTS=".:$(AAAI_KIT):" BSTINPUTS=".:$(AAAI_KIT):" latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex

clean:
	cd paper && latexmk -C main.tex && latexmk -C supplement.tex
	rm -rf experiments/__pycache__
