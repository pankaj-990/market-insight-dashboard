#!/usr/bin/env bash

set -euo pipefail

VENV_DIR=${VENV_DIR:-.venv}
REQ_FILE=${REQ_FILE:-requirements.txt}
PYTHON_BIN=${PYTHON_BIN:-python3}

echo "Resetting virtual environment at ${VENV_DIR}"
if [ -d "${VENV_DIR}" ]; then
  rm -rf "${VENV_DIR}"
fi

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

"${PYTHON_BIN}" -m pip install --upgrade pip

if [ -f "${REQ_FILE}" ]; then
  "${PYTHON_BIN}" -m pip install -r "${REQ_FILE}"
else
  echo "Requirements file '${REQ_FILE}' not found; skipping install."
fi

"${PYTHON_BIN}" -m pip install isort black ruff

echo "Running isort, black, and ruff..."
isort .
black .
ruff check . --fix

echo "Environment refresh complete."
