#!/usr/bin/bash
apt-get -y update && apt-get -y install gap   # needed for gappy
pip install poetry
# poetry config virtualenvs.in-project true
poetry install
poetry self add poetry-plugin-shell
poetry shell
