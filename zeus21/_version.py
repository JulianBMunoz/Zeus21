"""Single source of truth for the Zeus21 version.

The version is ``MAJOR.MINOR``:

* ``MAJOR`` is read from the ``VERSION`` file at the root of the repository. It is the
  only number a human ever edits. Right now it holds ``2``.
* ``MINOR`` is the number of commits made *since the commit that last changed*
  ``VERSION``. It is therefore ``0`` on the commit that sets a new major and grows by
  one with every commit after it. Bumping the major is just "edit VERSION, commit":
  the minor resets to 0 by itself.

Three situations, in this order of precedence:

* **inside a git work tree** (a clone, an editable install) the number is computed live,
  so ``zeus21.__version__`` follows your commits with no reinstall;
* **from an sdist or a wheel**, where there is no git history, the value that ``setup.py``
  froze into ``_static_version.txt`` when the distribution was built;
* failing both, the version pip recorded at install time.

Nothing here guesses. If none of the three is available the import fails loudly rather
than inventing a number.
"""

import shutil
import subprocess
from pathlib import Path

# zeus21/_version.py -> zeus21/ -> repository root
_ROOT = Path(__file__).resolve().parent.parent
_VERSION_FILE = _ROOT / "VERSION"

# written by setup.py at build time; the only thing an sdist can carry
_STATIC_FILE = Path(__file__).resolve().parent / "_static_version.txt"


def _git(*args):
    """Run git inside the repository and return stdout, or None if it did not work."""
    out = subprocess.run(["git", "-C", str(_ROOT), *args],
                         capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else None


def in_git_worktree():
    """True when this file lives inside a git checkout of zeus21 and git is usable."""
    if shutil.which("git") is None or not _VERSION_FILE.is_file():
        return False

    return _git("rev-parse", "--is-inside-work-tree") == "true"


def version_from_git():
    """MAJOR from VERSION, MINOR = first-parent commits since VERSION last changed."""
    major = _VERSION_FILE.read_text().strip()

    # the commit that last touched VERSION; empty on the very first commit that adds it
    anchor = _git("log", "-1", "--format=%H", "--", "VERSION")
    span = f"{anchor}..HEAD" if anchor else "HEAD"

    # --first-parent so that merging a branch adds one commit, not the whole branch
    minor = _git("rev-list", "--count", "--first-parent", span)

    return f"{major}.{minor}"


def get_version():
    if in_git_worktree():
        return version_from_git()

    if _STATIC_FILE.is_file():
        return _STATIC_FILE.read_text().strip()

    from importlib.metadata import version as _installed_version

    return _installed_version("zeus21")


def freeze(value):
    """Record `value` in _static_version.txt so a built distribution carries it.

    Called by setup.py. In a git checkout the live number always wins over this file,
    so a stale copy can never shadow your commits.
    """
    _STATIC_FILE.write_text(value + "\n")


__version__ = get_version()
