#!/usr/bin/env bash
# One-time developer setup. Run once after cloning:
#     bash tools/setup.sh
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Checking for a virtual environment"
if [ -z "${VIRTUAL_ENV:-}" ] && [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
    cat >&2 <<'MSG'
    !! No virtualenv active.

       Installing into your user site-packages can shadow system Python
       packages -- on a machine with ROS installed, a newer pytest breaks
       the launch_testing plugin, for example.

       Create one first:

           python3 -m venv .venv
           source .venv/bin/activate
           bash tools/setup.sh

MSG
    exit 1
fi
echo "    ${VIRTUAL_ENV:-$CONDA_DEFAULT_ENV}"

echo
echo "==> Installing git hooks"
git config core.hooksPath .githooks
echo "    hooksPath = .githooks (blocks direct pushes to main)"

echo
echo "==> Checking your git identity"
name="$(git config user.name  || true)"
email="$(git config user.email || true)"

if [ -z "$name" ] || [ -z "$email" ]; then
    echo "    !! Not set. Your commits will not be attributed to you."
    echo "       git config user.name  \"Your Full Name\""
    echo "       git config user.email \"your-github-email@example.com\""
    exit 1
fi

echo "    $name <$email>"
case "$email" in
    *@*.*) ;;
    *) echo "    !! '$email' has no top-level domain -- looks truncated."
       echo "       GitHub will not link these commits to your account."
       exit 1 ;;
esac

echo
echo "==> Installing the package"
cd src && pip install -e ".[dev]" --quiet
echo "    ok"

echo
echo "==> Running contract tests"
pytest -q

echo
echo "Setup complete. Your commits will be attributed to $name <$email>."
echo "Verify anyone's contribution count any time with:  git shortlog -sne"
