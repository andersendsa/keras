#!/bin/bash
/home/jules/.pyenv/versions/3.12.12/bin/python -m pytest -c ./pytest.ini ./keras/src/ops/linalg_test.py::LinalgOpsCorrectnessTest::test_qr
