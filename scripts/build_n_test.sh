#!/bin/sh
set -e

flake8 --config=code/flake8.cfg
python -m unittest discover