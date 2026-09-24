"""Every path a script or module derives from its own location must still resolve.

This exists because moving files is the one refactor that breaks paths silently. `robot/demo.sh` moved from
the repository root into `robot/` and kept `REPO="$(cd "$(dirname "$0")" && pwd)"`, which then pointed at
`robot/` instead of the root, so the very first command it ran was a python interpreter that does not exist.
Three module-level defaults in `training/` broke the same way at the same time, and the whole suite stayed
green: nothing asserted that a default path points at something real.

`--help` would not have caught it either -- these scripts compute their paths before parsing arguments.
"""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# (module, attribute): a path the module computes from __file__ when imported.
MODULE_PATHS = [
    ("core.motion.detectors", "MODELS_DIR"),
    ("core.motion.detectors", "DEFAULT_MODEL_PATH"),
    ("core.motion.detectors", "DEFAULT_FOREST_PATH"),
    ("core.motion.actions.detector", "MODELS_DIR"),
    ("core.motion.actions.detector", "DEFAULT_ACTION_MODEL"),
    ("training.dataset", "DATA_DIR"),
    ("training.train_public", "DEFAULT_OUT"),
    ("training.actions.train", "MODELS_DIR"),
]


@pytest.mark.parametrize("module_name,attribute", MODULE_PATHS)
def test_module_default_path_resolves(module_name, attribute):
    """A default that points outside the repository, or at a missing directory, is a moved-file bug.

    A `*_DIR` attribute is asserted to be a directory itself. For a file default (a trained model that may
    not have been produced yet) the assertion is on its containing directory instead. The distinction
    matters: falling back to the parent for a directory attribute would compare ROOT/"models" against its
    parent ROOT, which always exists -- silently passing the exact bug this catches.
    """
    import importlib

    path = getattr(importlib.import_module(module_name), attribute)
    directory = path if attribute.endswith("_DIR") else path.parent
    assert ROOT in directory.parents or directory == ROOT, \
        f"{module_name}.{attribute} = {path} escapes the repository"
    assert directory.is_dir(), f"{module_name}.{attribute} = {path}: {directory} does not exist"


def shell_scripts_deriving_a_root():
    """Tracked shell scripts that compute a base directory from their own location."""
    for script in sorted(ROOT.rglob("*.sh")):
        if "reachy_mini_env" in script.parts:
            continue
        for line in script.read_text().splitlines():
            if re.match(r'^[A-Z_]+="\$\(cd "\$\(dirname "\$0"\)', line):
                yield script, line


@pytest.mark.parametrize("script,line", list(shell_scripts_deriving_a_root()),
                         ids=lambda v: v.name if isinstance(v, Path) else "")
def test_shell_script_base_directory_is_the_repo_root(script, line):
    """Evaluate the script's own assignment with $0 set to the script, and check where it lands.

    The right-hand side is taken verbatim from the file and run by bash with $0 bound to that script, so
    this measures what the script itself would compute rather than re-deriving it independently.
    """
    name, _, expression = line.partition("=")
    # Evaluate ONLY the `$(cd ... && pwd)` substitution, dropping anything the script appends to it
    # (download_datasets.sh adds `/datasets`). Accepting an appended suffix would make this assertion
    # tolerate a root that is one directory too deep -- which is precisely the bug it exists to catch.
    # Non-greedy up to `&& pwd)`: a character class excluding ")" would stop at the nested $(dirname ...).
    substitution = re.search(r'\$\(cd .*?&&\s*pwd\)', expression)
    assert substitution, f"could not isolate the cd expression in {script.name}: {expression}"
    result = subprocess.run(
        ["bash", "-c", f'printf "%s" "{substitution.group(0)}"', str(script)],
        capture_output=True, text=True, check=True,
    )
    base = Path(result.stdout)
    assert base == ROOT, \
        f"{script.relative_to(ROOT)} derives {name} from {base}, expected the repo root {ROOT}"


@pytest.mark.parametrize("script", sorted({s for s, _ in shell_scripts_deriving_a_root()}),
                         ids=lambda p: p.name)
def test_shell_script_references_under_the_root_exist(script):
    """Each "$REPO/<path>" a script names must exist, which is what a broken root makes false."""
    text = script.read_text()
    missing = [ref for ref in sorted(set(re.findall(r'\$(?:REPO|DIR)/([A-Za-z0-9_./-]+)', text)))
               if not (ROOT / ref).exists()]
    assert not missing, f"{script.relative_to(ROOT)} references paths that do not exist: {missing}"
