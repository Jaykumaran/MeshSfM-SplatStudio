#!/usr/bin/env bash
# Creates ./.venv and installs MeshSfM SplatStudio with the LiDAR and mask extras.
# COLMAP (with CUDA) must be on PATH for the RGB pipeline: https://colmap.github.io/install.html
set -e
cd "$(dirname "$(readlink -f "$0")")"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[lidar,masks]"
echo "done. Start with:  ./meshsfm.sh ~/meshsfm_projects/demo"
