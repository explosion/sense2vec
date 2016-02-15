from __future__ import print_function
import sys

import plac
import sputnik
from sputnik.package_list import (PackageNotFoundException,
                                  CompatiblePackageNotFoundException)

from sense2vec import about


@plac.annotations(
    force=("Force overwrite", "flag", "f", bool),
)
def main(force=False):
    if force:
        sputnik.purge(about.__title__, about.__version__)

    try:
        sputnik.package(about.__title__, about.__version__, about.__default_model__)
        print("Model already installed. Please run '%s --force to reinstall." % sys.argv[0], file=sys.stderr)
        sys.exit(1)
    except (PackageNotFoundException, CompatiblePackageNotFoundException):
        pass

    package = sputnik.install(about.__title__, about.__version__, about.__default_model__)

    try:
        sputnik.package(about.__title__, about.__version__, about.__default_model__)
    except (PackageNotFoundException, CompatiblePackageNotFoundException):
        print("Model failed to install. Please run '%s --force." % sys.argv[0], file=sys.stderr)
        sys.exit(1)

    print("Model successfully installed.", file=sys.stderr)


if __name__ == '__main__':
    plac.call(main)
