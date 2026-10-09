#!/usr/bin/env python3
"""Lay this repository over a checkout of KingKongRobotics/jumper.

    python tools/install_into_jumper.py /path/to/jumper            # install
    python tools/install_into_jumper.py /path/to/jumper --check    # report only

Copies the files that live at the same path in both trees (the trampoline
scene, the bed calibration tool, the bouncing task) and adds the two
registration lines the toolkit needs: the scene's `register(...)` block in
`scenes/__init__.py` and the task's import line in `tasks/__init__.py`. Both
edits are idempotent, so running it again after a toolkit update only refreshes
the files. Needs no dependency beyond Python.
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent

#: Files and directories copied verbatim, relative to both roots.
COPIES = (
    "scenes/trampoline.py",
    "tools/calibrate_bed.py",
    "tasks/jumper/bounce",
)

SCENE_REGISTER = '''register(
    id="trampoline",
    description="**changes the ground**: a trampoline bed as under-damped soft contact; "
                "`--bed-dampratio` is the bounce, `--bed-timeconst` the depth",
    tags=("physics", "ground"),
    use="both",
)
'''
#: Inserted just before this block in scenes/__init__.py.
SCENE_ANCHOR = 'register(\n    id="plain",'

TASK_IMPORT = "from .jumper import bounce  # noqa: F401,E402\n"
#: Inserted just before this line in tasks/__init__.py (the imports are sorted).
TASK_ANCHOR = "from .jumper import dance  # noqa: F401,E402\n"


def _same(a: Path, b: Path) -> bool:
    if a.is_dir():
        if not b.is_dir():
            return False
        cmp = filecmp.dircmp(a, b, ignore=["__pycache__"])
        return not (cmp.left_only or cmp.right_only or cmp.diff_files or cmp.funny_files) and all(
            _same(a / d, b / d) for d in cmp.common_dirs
        )
    return b.is_file() and filecmp.cmp(a, b, shallow=False)


def _copy(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _insert(path: Path, needle: str, anchor: str, block: str, check: bool) -> str:
    text = path.read_text(encoding="utf-8")
    if needle in text:
        return "present"
    if anchor not in text:
        return f"ANCHOR NOT FOUND ({anchor.splitlines()[0]!r}); add the block by hand"
    if not check:
        path.write_text(text.replace(anchor, block + anchor, 1), encoding="utf-8")
    return "inserted" if not check else "missing"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jumper", type=Path, help="path to a checkout of KingKongRobotics/jumper")
    p.add_argument("--check", action="store_true", help="report what differs, change nothing")
    a = p.parse_args()
    root = a.jumper.resolve()
    for must in ("scenes/__init__.py", "tasks/__init__.py", "scripts/train.py"):
        if not (root / must).is_file():
            sys.exit(f"{root} does not look like a jumper checkout: no {must}")

    problems = 0
    for rel in COPIES:
        src, dst = HERE / rel, root / rel
        if _same(src, dst):
            print(f"  same      {rel}")
        elif a.check:
            print(f"  DIFFERS   {rel}")
            problems += 1
        else:
            _copy(src, dst)
            print(f"  copied    {rel}")

    r = _insert(root / "scenes/__init__.py", 'id="trampoline"', SCENE_ANCHOR, SCENE_REGISTER, a.check)
    print(f"  {r:<9} scenes/__init__.py: trampoline register block")
    problems += r not in ("present", "inserted")
    r = _insert(root / "tasks/__init__.py", TASK_IMPORT, TASK_ANCHOR, TASK_IMPORT, a.check)
    print(f"  {r:<9} tasks/__init__.py: bounce import line")
    problems += r not in ("present", "inserted")

    if not a.check:
        print("\nNext: python scripts/train.py --list   (expect jumper.bounce and the trampoline scene)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
