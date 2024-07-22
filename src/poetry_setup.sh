#!/usr/bin/bash
pip install poetry
poetry config virtualenvs.in-project true
poetry install
poetry shell
