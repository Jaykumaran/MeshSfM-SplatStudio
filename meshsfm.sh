#!/usr/bin/env bash
# MeshSfM SplatStudio launcher
#   ./meshsfm.sh PROJECT_DIR              -> open the studio at http://127.0.0.1:8780 (creates the project if new)
#   ./meshsfm.sh <command> ...            -> command line (add / prepare / gen / align / kitti), see README
# Python environment: $MESHSFM_VENV if set, else ./.venv if it exists, else the python on PATH.
set -e
cd "$(dirname "$(readlink -f "$0")")"
if [ -n "${MESHSFM_VENV:-}" ]; then source "$MESHSFM_VENV/bin/activate"
elif [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
export PYTHONUNBUFFERED=1
case "${1:-}" in
  ""|-h|--help) exec python -m meshsfm --help ;;
  serve|add|prepare|gen|align|kitti) exec python -m meshsfm "$@" ;;
  *) exec python -m meshsfm serve "$@" ;;
esac
