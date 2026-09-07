#!/bin/sh
set -eu

PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=. \
  python -m p04.verify_release_bundle .
