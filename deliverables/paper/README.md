# Paper

| File | What it is |
| --- | --- |
| `main.tex` | The submitted paper: 2 pages, IEEE conference format (`IEEEtran`), one overview figure and one comparison table |
| `main_extended.tex` | The extended version: every experiment with its own table or figure, the full setup and the learning-rate sweep |

Every number in both comes from [../../results/](../../results/); [../../results/README.md](../../results/README.md)
says which file.

**Compile on Overleaf:** upload the `.tex` file, make a folder `figures/` next to it, and upload the PNG files it
uses from [../../results/figures/](../../results/figures/) (`main.tex` needs only `fig10_overview.png`). Compile
with pdfLaTeX.

**Compile locally:** run `pdflatex main.tex` twice in this folder; the figures are found in
`../../results/figures/`.

The final PDF is [AFML_Abcd_paper.pdf](AFML_Abcd_paper.pdf), compiled from `main.tex` on Overleaf.
