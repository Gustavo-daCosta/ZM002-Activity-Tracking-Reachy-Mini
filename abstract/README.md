### Using Dutch/English

By default, the template uses English, however you can switch to Dutch simply
by adding the `dutch` option to the `main.tex` `\documentclass`
definition. For example:

```tex
% default
\documentclass{adsai}
```

```tex
% in Dutch
\documentclass[dutch]{adsai}
```

### Using a different bibliography style

By default, the template uses IEEE-style bibliography, if you want to use APA7,
MLA or something else, simply add it to the `\documentclass` options. For
example:

```tex
% default
\documentclass{adsai}
```

```tex
% using APA7
\documentclass[bibstyle=apa]{adsai}
```

```tex
% both APA7 and Dutch
\documentclass[bibstyle=apa, dutch]{adsai}
```

### Rendering the document

From the repository root, run:

```bash
python abstract/render.py
```

This runs `latexmk` on `main.tex` and writes the rendered document to
`abstract/main.pdf`. The `-g` option forces a rebuild, which is useful after
fixing a previous LaTeX error. `latexmk` automatically runs the required
LaTeX and `biber` passes for the bibliography. To write a copy somewhere else,
use:

```bash
python abstract/render.py --output /path/to/abstract.pdf
```

The required `latexmk` command and LaTeX packages are included in common
LaTeX distributions. On Arch Linux, install the packages used by this
template with:

```bash
sudo pacman -S texlive-latex texlive-latexextra texlive-pictures \
  texlive-bibtexextra texlive-publishers texlive-fontsextra \
  texlive-mathscience biber
```
