import pathlib
import sys

root_dir = pathlib.Path(__file__).resolve().parent.parent
# Ensure project root is on Python path for module resolution (dag_logger, dag_model, etc.)
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Also make the tests directory itself first for local helper imports
tests_dir = pathlib.Path(__file__).resolve().parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

from test_common import *  # re-export fixtures
