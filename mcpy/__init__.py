import logging
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version('mcpy')
except PackageNotFoundError:  # not installed, e.g. run from a source checkout
    __version__ = '0.0.0.unknown'

# Library convention: attach a NullHandler to the top-level package logger so
# importing mcpy has no side effects on the host application's logging config.
# Users opt in to logging via mcpy.utils.logging.configure(...).
logging.getLogger(__name__).addHandler(logging.NullHandler())
