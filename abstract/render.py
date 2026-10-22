"""Compile the abstract LaTeX document into a PDF with latexmk."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


ABSTRACT_DIR = Path(__file__).resolve().parent
MAIN_TEX = ABSTRACT_DIR / "main.tex"


def render(output: Path | None = None) -> Path:
    """Render ``main.tex`` and return the generated PDF path."""
    latexmk = shutil.which("latexmk")
    if latexmk is None:
        raise RuntimeError(
            "latexmk was not found. Install a LaTeX distribution first."
        )

    if shutil.which("biber") is None:
        raise RuntimeError(
            "biber was not found. This document uses biblatex; on Arch Linux "
            "install it with: sudo pacman -S biber"
        )

    if not MAIN_TEX.is_file():
        raise FileNotFoundError(f"Main LaTeX file not found: {MAIN_TEX}")

    subprocess.run(
        [
            latexmk,
            "-pdf",
            "-g",
            "-interaction=nonstopmode",
            "-halt-on-error",
            MAIN_TEX.name,
        ],
        cwd=ABSTRACT_DIR,
        check=True,
    )

    generated_pdf = MAIN_TEX.with_suffix(".pdf")
    if output is not None:
        output = output.resolve()
        if output != generated_pdf.resolve():
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(generated_pdf, output)
        return output

    return generated_pdf


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render abstract/main.tex as a PDF."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional path for a copy of the rendered PDF.",
    )
    args = parser.parse_args()

    try:
        pdf_path = render(args.output)
    except RuntimeError as error:
        parser.error(str(error))

    print(f"Rendered PDF: {pdf_path}")


if __name__ == "__main__":
    main()
