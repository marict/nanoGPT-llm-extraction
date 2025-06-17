import sys

MIN_VERSION = (3, 10)


def check_python_version() -> None:
    """Raise ``RuntimeError`` if the running Python is too old."""
    if sys.version_info < MIN_VERSION:
        required = ".".join(map(str, MIN_VERSION))
        raise RuntimeError(
            f"Python {required}+ is required, found {sys.version.split()[0]}"
        )

