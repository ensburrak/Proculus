"""Compatibility wrapper around the canonical ML dataset builder."""

from ml.build_dataset import *  # noqa: F401,F403


if __name__ == "__main__":
    from ml.build_dataset import main

    main()
