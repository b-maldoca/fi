"""Pytest root configuration.

This file is intentionally empty of fixtures. Its presence at the repository
root is what makes ``from src...`` imports work in the test suite: in pytest's
default "prepend" import mode, the rootdir containing the topmost ``conftest.py``
is inserted into ``sys.path``. Without it, pytest would only insert ``tests/``
(which has no ``__init__.py``) and the ``src`` package would not be importable.
"""
