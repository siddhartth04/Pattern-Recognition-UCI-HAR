# IEEE LaTeX manuscript

`main.tex` is the complete source. `references.bib` contains 17 references, all cited. `generated/` contains result-table rows and `figures/robustness.pdf` is a vector scientific plot. `main.pdf` is the compiled five-page paper, identical to `../paper/final_paper.pdf`.

Compile in this folder with `tectonic main.tex`, or `pdflatex main.tex`, `bibtex main`, and `pdflatex main.tex` twice. IEEEtran must be available. Tectonic downloads its required TeX resources on first use. `main.bbl` is included for convenient source submission, but the bibliography remains editable.

To regenerate tables and figure, run `python -m src.revision_report` from the implementation folder. Edit narrative prose in `main.tex`; the report generator only regenerates tables and the plot.

This is generic IEEE conference formatting on US Letter, not certification against a named venue. The author field is intentionally anonymous. Resolve author information, page limits, anonymity, disclosure and any required PDF-validation checks before submitting.
