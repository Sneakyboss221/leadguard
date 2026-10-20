# Paper

IEEE two-column conference paper (`IEEEtran`, conference mode). Main file: `main.tex`; one file per section in `sections/`.

## Numbers
No number is typed by hand. `make_numbers.py` reads `results/*.csv` and writes:
- `generated/numbers.tex`: one `\newcommand` per number, each line commented with the CSV it comes from;
- `generated/tab_*.tex`: the tables;
- `figures/`: copies of the result figures.

Re-run after any change to the results:
```
python paper/make_numbers.py
```

## Compile
- Overleaf: upload the whole `paper/` folder, set `main.tex` as the main document.
- Locally: `pdflatex main.tex` (twice), once a TeX distribution such as MiKTeX is installed.

Red `[TODO: ...]` and `[CITE: ...]` markers show what is unfinished; the final PDF must contain none. References go into
`refs.bib` only after each one has been checked against its primary source.
