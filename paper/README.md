# Paper

IEEE two-column journal paper (`IEEEtran`, journal mode), prepared for IEEE JBHI. Main file: `main.tex`; one file
per section in `sections/`; references in `refs.bib`, each checked against its primary source.

## Numbers and figures
No number is typed by hand. `make_numbers.py` reads `../results/*.csv` and writes:
- `generated/numbers.tex`: one `\newcommand` per number, each line commented with the CSV it comes from;
- `generated/tab_*.tex`: the tables;
- `figures/fig_*.pdf`: the vector figures.

`facts.tex` holds the few numbers that do not come from a CSV (dataset facts), each with its source.
`make_vignette.py` draws the single-patient figure by running the trained models through the demo code, so it needs
the checkpoints from the v1.0 release.

```
python paper/make_numbers.py
python paper/make_vignette.py
```

## Compile
- Overleaf: upload the whole `paper/` folder, set `main.tex` as the main document.
- Locally: `pdflatex main`, `bibtex main`, then `pdflatex main` twice.
