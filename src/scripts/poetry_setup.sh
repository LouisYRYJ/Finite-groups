#!/usr/bin/bash
apt-get update && apt-get install gap   # needed for gappy
pip install poetry
# poetry config virtualenvs.in-project true
poetry install
poetry shell
