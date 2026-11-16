"""Generate the reproducible Pancake/TopSpin development collection.

Run from an installed source checkout:
  python examples/generate_permutation_collection.py --output artifacts/permutation-v1
Equivalent to `copuzzle permutation collection` (see --help for options).
"""
import sys

from copuzzle.games.permutation.cli import main


if __name__ == "__main__":
    main(["collection", *sys.argv[1:]])
