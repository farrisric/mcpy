import logging

# Library convention: attach a NullHandler to the top-level package logger so
# importing mcpy has no side effects on the host application's logging config.
# Users opt in to logging via mcpy.utils.logging.configure(...).
logging.getLogger(__name__).addHandler(logging.NullHandler())
