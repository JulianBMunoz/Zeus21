"""Single source of truth for the package version.

Identical file for zeus21 and oLIMpus: it takes the distribution name from the directory
it sits in, so drop it in as ``<package>/_version.py`` and nothing else changes.

The version is ``MAJOR.MINOR``:

* ``MAJOR`` is read from the ``VERSION`` file at the root of the repository. It is the
  only number a human ever edits, and it must be committed.
* ``MINOR`` is the number of commits made *since the commit that last changed*
  ``VERSION``. It is therefore ``0`` on the commit that sets a new major and grows by
  one with every commit after it. Bumping the major is just "edit VERSION, commit":
  the minor resets to 0 by itself.

Three situations, in this order of precedence:

* **inside a checkout of this repository** (a clone, an editable install) the number is
  computed live, so ``<package>.__version__`` follows your commits with no reinstall;
* **from an sdist or a wheel**, where there is no git history, the value that
  ``setup.py`` froze into ``_static_version.txt`` when the distribution was built;
* failing both, the version pip recorded at install time.

Nothing here guesses. Inside a checkout with no ``VERSION`` file the import fails and
says what to create, rather than quietly serving a frozen number that never moves.
"""

import shutil
import subprocess
from pathlib import Path

_HERE = Path(__file__).resolve().parent      # <repo>/<package>
_ROOT = _HERE.parent                         # <repo>
_PACKAGE = _HERE.name                        # "zeus21" or "oLIMpus"
_VERSION_FILE = _ROOT / "VERSION"

# written by setup.py at build time; the only thing an sdist can carry.
# It is a build artefact: keep it in .gitignore, never commit it.
_STATIC_FILE = _HERE / "_static_version.txt"


def _git(*args):
    """Run git inside the repository and return stdout, or None if it did not work."""
    out = subprocess.run(["git", "-C", str(_ROOT), *args],
                         capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else None


def in_git_worktree():
    """True when this file sits in a checkout of THIS repository.

    Compares against the work tree's top level rather than merely asking "is this inside
    some git repo": a wheel installed into a site-packages directory that happens to live
    under an unrelated repository must not be versioned from that repository's history.
    """
    if shutil.which("git") is None:
        return False

    top = _git("rev-parse", "--show-toplevel")

    return top is not None and Path(top).resolve() == _ROOT


def version_from_git():
    """MAJOR from VERSION, MINOR = first-parent commits since VERSION last changed."""
    if not _VERSION_FILE.is_file():
        raise FileNotFoundError(
            f"{_VERSION_FILE} does not exist. It holds the major version on a single "
            f"line and the minor is counted from the commit that last changed it, so "
            f"the version cannot be derived without it. Create and commit it:\n"
            f"    echo 2 > {_VERSION_FILE}\n"
            f"    git add VERSION && git commit -m 'set major version'")

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

    return _installed_version(_PACKAGE)


def freeze(value):
    """Record `value` in _static_version.txt so a built distribution carries it.

    setup.py must call this, otherwise every sdist and wheel ships whatever stale value
    happens to be on disk. In a checkout the live number always wins over this file, so
    a stale copy can never shadow your commits.
    """
    _STATIC_FILE.write_text(value + "\n")


__version__ = get_version()