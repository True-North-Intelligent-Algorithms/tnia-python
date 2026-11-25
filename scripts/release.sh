#!/bin/bash
# Build and publish tnia-python to PyPI.
#
#   scripts/release.sh            # pyproject.toml's version; if PyPI has it, +1 (0.2.3 -> 0.2.4)
#   scripts/release.sh 0.3.0      # or name the version
#
# Needs only uv; no special environment (uv build makes its own).
# PyPI token: set UV_PUBLISH_TOKEN, or uv publish asks (username __token__).
# Sets the version in pyproject.toml, commits, tags v<version>, builds,
# uploads, then pushes main and the tag.
set -euo pipefail
cd "$(dirname "$0")/.."

version=${1:-}

# Only the package has to be committed; notebooks don't go into it.
if [ -n "$(git status --porcelain -- tnia MANIFEST.in README_pypi.md)" ]; then
    echo "Uncommitted changes in the package. Commit first." >&2
    exit 1
fi
# pyproject.toml may differ only in its version line (this script sets it).
if git diff -U0 -- pyproject.toml | grep '^[-+][^-+]' | grep -qv '^[-+]version = '; then
    echo "Uncommitted changes in pyproject.toml. Commit first." >&2
    exit 1
fi

on_pypi() { curl -sfo /dev/null "https://pypi.org/pypi/tnia-python/$1/json"; }
latest=$(curl -sf https://pypi.org/pypi/tnia-python/json |
    python3 -c 'import json, sys; print(json.load(sys.stdin)["info"]["version"])')
current=$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)

if [ -z "$version" ]; then
    version=$current
    if on_pypi "$version"; then
        if [[ ! $version =~ ^([0-9]+\.[0-9]+\.)([0-9]+)$ ]]; then
            echo "Can't bump $version. Pass the version." >&2
            exit 1
        fi
        version="${BASH_REMATCH[1]}$((BASH_REMATCH[2] + 1))"
    fi
fi

if on_pypi "$version"; then
    echo "$version is already on PyPI. Pass a newer version." >&2
    exit 1
fi

echo "PyPI has $latest. Release $version from $(git log --oneline -1)"
read -rp "Commit, tag, build, upload, push? [y/N] " ok
[ "$ok" = y ] || exit 1

sed -i "s/^version = \".*\"/version = \"$version\"/" pyproject.toml
git diff --quiet -- pyproject.toml || git commit -m "Release $version" -- pyproject.toml
git tag "v$version"

uv build --out-dir dist
uv publish "dist/tnia_python-$version"*

git push origin main "v$version"
echo "Done. pip install tnia-python==$version"
